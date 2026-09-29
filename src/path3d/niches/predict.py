"""Xenium tissue-neighbourhood prediction from H&E, as an importable module.

This is ``predict_niches.py`` (Cook Lab, ``2026_final_xenium_analysis``)
restructured into functions so :mod:`path3d` can call it per section instead of
shelling out once per slide. **The numerics are unchanged** -- the tile grid,
the Macenko normalisation, the 224x224 bilinear resize, the UNI2-h forward
pass and the per-slide quantile calls all reproduce the original script
exactly. See ``docs/niches/MODEL_CARD.md`` for validation and limits.

Four classes: ``acellular | epithelium | immune | stroma`` (the secretory
niches are deliberately collapsed into one ``epithelium`` class -- the
SecA->SecB axis is not recoverable from morphology).

The original CLI is preserved::

    python -m path3d.niches.predict SLIDE.ome.tiff --out results/

with one deliberate divergence: ``--mpp`` is now **optional**. When omitted it
is read from the image's own OME-XML / TIFF resolution tags, which is strictly
safer than a human retyping it -- a wrong ``--mpp`` silently changes the
physical size of every tile and degrades everything, and is the single easiest
way to get meaningless output. Pass ``--mpp`` only to override a file whose
metadata you know to be wrong.

Requires Hugging Face access to the gated ``MahmoodLab/UNI2-h``: request at
<https://huggingface.co/MahmoodLab/UNI2-h>, then ``hf auth login``. First run
downloads ~2.5 GB of weights and caches them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from path3d.niches.stain import macenko

# Packaged classifier. Paths are resolved relative to this file so the package
# stays relocatable -- never hardcode an absolute path here.
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "niche4.joblib"

# Fallbacks only. The real values are read from the model bundle's own
# ``tile_um`` / ``fov_um`` keys so the grid can never drift from what the
# classifier was trained on.
TILE_UM, FOV_UM, REPORT_UM = 32.0, 112.0, 250.0

BLOCK, BATCH = 4096, 32
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# ImageNet-style input the UNI2-h checkpoint expects.
_EMBED_PX = 224
_EMBED_DIM = 1536

# Bundles whose sklearn-version mismatch has already been reported, so a long
# run that loads the model more than once says it once.
_VERSION_WARNED: set[Path] = set()


def log(m: str) -> None:
    print(m, flush=True)


# --------------------------------------------------------------------------
# model bundle
# --------------------------------------------------------------------------


@dataclass
class NicheModel:
    """A loaded ``niche4.joblib`` bundle.

    Attributes:
        clf: the fitted scikit-learn classifier (``predict_proba`` on
            L2-normalised UNI2-h embeddings).
        classes: class names in ``clf.classes_`` order -- the column order of
            every probability array this module produces.
        tile_um: side of the prediction grid cell, microns.
        fov_um: side of the image window fed to UNI2-h, microns.
        auc_by_class: leave-one-tumour-out AUC per class, for reporting.
        scheme, trained_on: provenance strings from the bundle.
    """

    clf: Any
    classes: list[str]
    tile_um: float
    fov_um: float
    auc_by_class: dict[str, float] = field(default_factory=dict)
    scheme: str = ""
    trained_on: str = ""


def load_model(path: str | Path | None = None) -> NicheModel:
    """Load the niche classifier bundle.

    Args:
        path: bundle to load; defaults to the packaged ``niche4.joblib``.

    Returns:
        A :class:`NicheModel`.

    Security:
        ``joblib.load`` unpickles, which **executes arbitrary code**. Only ever
        point this at the packaged bundle or one you produced yourself -- never
        at a file from an untrusted source.
    """
    import joblib

    path = Path(path) if path is not None else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Niche model bundle not found: {path}. The packaged bundle should "
            f"live at {DEFAULT_MODEL_PATH}."
        )

    # sklearn's own InconsistentVersionWarning fires from deep inside the
    # unpickle with no context about which model or why it matters. Catch it
    # and re-raise something actionable, once, naming both versions -- this
    # would otherwise surface as a cryptic line in the middle of a multi-hour
    # stack run. tests/test_niches_model.py asserts the estimator still
    # computes the correct closed-form result under the installed sklearn.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = joblib.load(path)
    for record in caught:
        if "InconsistentVersionWarning" in record.category.__name__:
            import sklearn

            if path in _VERSION_WARNED:
                continue
            _VERSION_WARNED.add(path)
            warnings.warn(
                f"{path.name} was pickled with scikit-learn "
                f"{getattr(record.message, 'original_sklearn_version', 'unknown')} "
                f"but scikit-learn {sklearn.__version__} is installed. The "
                f"bundled classifier is a plain LogisticRegression, whose "
                f"coef_/intercept_ carry all the fitted state, so this is "
                f"normally harmless -- but verify before trusting a run: "
                f"pytest tests/test_niches_model.py",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            warnings.warn_explicit(
                record.message,
                record.category,
                record.filename,
                record.lineno,
            )

    clf = bundle["model"]
    return NicheModel(
        clf=clf,
        classes=list(clf.classes_),
        tile_um=float(bundle.get("tile_um", TILE_UM)),
        fov_um=float(bundle.get("fov_um", FOV_UM)),
        auc_by_class=dict(bundle.get("loso_auc_by_class", {})),
        scheme=str(bundle.get("scheme", "")),
        trained_on=str(bundle.get("trained_on", "")),
    )


def prob_columns(classes: list[str]) -> list[str]:
    """``p_<class>`` column names, in ``classes`` order."""
    return [f"p_{c.replace(' ', '_')}" for c in classes]


def call_columns(classes: list[str]) -> list[str]:
    """``call_<class>`` column names, in ``classes`` order."""
    return [f"call_{c.replace(' ', '_')}" for c in classes]


# --------------------------------------------------------------------------
# slide metadata
# --------------------------------------------------------------------------


def read_mpp(slide_path: str | Path) -> float:
    """Read level-0 microns-per-pixel from a TIFF's own metadata.

    Prefers OME-XML ``PhysicalSizeX`` (what
    ``registration.warp_and_save_section`` writes via
    ``update_xml_for_new_img``), then falls back to the baseline
    ``XResolution``/``ResolutionUnit`` tags (what it stamps via
    ``copy(xres=1000.0 / target_mpp)``, vips resolution being px/mm).

    Args:
        slide_path: path to a TIFF / OME-TIFF.

    Returns:
        Microns per pixel at level 0.

    Raises:
        ValueError: neither source yields a usable physical pixel size. Pass
            ``--mpp`` explicitly in that case -- do not guess.
    """
    import tifffile

    with tifffile.TiffFile(str(slide_path)) as tf:
        # 1. OME-XML PhysicalSizeX
        if tf.is_ome and tf.ome_metadata:
            try:
                import xml.etree.ElementTree as ET

                root = ET.fromstring(tf.ome_metadata)
                for pixels in root.iter():
                    if not pixels.tag.endswith("Pixels"):
                        continue
                    size = pixels.get("PhysicalSizeX")
                    if size is None:
                        continue
                    unit = (pixels.get("PhysicalSizeXUnit") or "µm").strip()
                    value = float(size)
                    if value <= 0:
                        continue
                    # Both micro-sign variants appear in the wild: U+00B5
                    # MICRO SIGN and U+03BC GREEK SMALL LETTER MU.
                    if unit in ("µm", "um", "micron", "microns", "μm"):
                        return value
                    if unit == "nm":
                        return value / 1000.0
                    if unit == "mm":
                        return value * 1000.0
            except (ET.ParseError, ValueError, TypeError):
                pass

        # 2. Baseline TIFF resolution tags
        page = tf.series[0].pages[0]
        xres = page.tags.get("XResolution")
        unit_tag = page.tags.get("ResolutionUnit")
        if xres is not None and xres.value:
            num, den = xres.value
            if num and den:
                px_per_unit = float(num) / float(den)
                if px_per_unit > 0:
                    unit = int(getattr(unit_tag, "value", 2))
                    if unit == 3:  # centimetre
                        return 10_000.0 / px_per_unit
                    if unit == 2:  # inch
                        return 25_400.0 / px_per_unit

    raise ValueError(
        f"Could not read a physical pixel size from {slide_path}. Pass --mpp "
        f"explicitly with the microns-per-pixel of YOUR level-0 image -- a "
        f"wrong value silently degrades every tile."
    )


def canvas_wh(slide_path: str | Path) -> tuple[int, int]:
    """Level-0 ``(width, height)`` in pixels.

    This is the canvas every niche grid for this section is defined on. All
    sections warped by ``registration.warp_and_save_section`` from one VALIS
    run share it, which is what lets :func:`path3d.niches.volume.build_niche_volume`
    stack them.
    """
    import tifffile

    with tifffile.TiffFile(str(slide_path)) as tf:
        h, w = tf.series[0].levels[0].shape[:2]
    return int(w), int(h)


# --------------------------------------------------------------------------
# tissue detection + tile grid  (numerics preserved from predict_niches.py)
# --------------------------------------------------------------------------


def tissue_mask(img: np.ndarray, min_component_frac: float = 0.002) -> np.ndarray:
    """Image-only tissue detection: dark AND saturated, then drop small islands.

    The training pipeline used spatial-transcriptomics cell positions to find
    tissue; that is not available here, so tissue comes from the image alone.
    Requiring SATURATION as well as darkness is what excludes grey scanner
    background and grey ink/fiducial marks, which a darkness-only threshold
    keeps.

    On a **registered** section this also rejects VALIS's canvas fill: the
    warp pads with near-white, which fails the ``sat > 0.12`` test.

    Vendored unchanged from ``predict_niches.py``.
    """
    from scipy import ndimage as ndi

    rgb = img[..., :3].astype(np.float32)
    gray = rgb.mean(2)
    mx = rgb.max(2)
    mn = rgb.min(2)
    sat = np.divide(mx - mn, np.maximum(mx, 1e-6))
    m = ndi.binary_fill_holes((gray < 225) & (sat > 0.12))
    m = ndi.binary_opening(m, np.ones((5, 5)))
    lab, n = ndi.label(m)
    if n:
        sizes = np.bincount(lab.ravel())
        keep = np.where(sizes >= max(50, min_component_frac * m.size))[0]
        m = np.isin(lab, keep[keep > 0])
    return m


def grid_shape(canvas_wh_px: tuple[int, int], mpp: float, tile_um: float) -> tuple[int, int]:
    """``(ny, nx)`` of the niche grid for a canvas of ``canvas_wh_px``.

    The single source of truth for the grid geometry. :func:`tile_grid` and
    :func:`path3d.niches.rasterize.rasterize_tiles` both call it, so a tile's
    ``(cx_px, cy_px)`` and its grid cell can never disagree.

    Args:
        canvas_wh_px: level-0 ``(width, height)`` in pixels.
        mpp: level-0 microns per pixel.
        tile_um: grid cell side in microns (``NicheModel.tile_um``).

    Returns:
        ``(ny, nx)`` -- rows, columns.
    """
    w0, h0 = canvas_wh_px
    step = tile_um / mpp
    return int(h0 // step), int(w0 // step)


def tile_grid(
    canvas_wh_px: tuple[int, int],
    mpp: float,
    tissue: np.ndarray,
    *,
    tile_um: float = TILE_UM,
    fov_um: float = FOV_UM,
    stride: int = 1,
) -> pd.DataFrame:
    """Build the on-tissue tile table for one section.

    Args:
        canvas_wh_px: level-0 ``(width, height)`` in pixels.
        mpp: level-0 microns per pixel.
        tissue: boolean tissue mask, at any resolution -- its shape relative to
            ``canvas_wh_px`` sets the scale factor, exactly as the original
            script derived ``scale`` from the mask pyramid level.
        tile_um: grid cell side, microns.
        fov_um: image window side fed to UNI2-h, microns.
        stride: keep every ``stride``-th grid cell in each axis, for a fast
            full-extent preview at 1/stride^2 the cost. Tiles stay on their
            original grid positions, so the result rasterises correctly -- it
            is simply sparser, and ``rasterize_tiles`` leaves the skipped cells
            NaN. ``1`` (the default) predicts every tile.

    Returns:
        DataFrame with ``cx_px, cy_px, x0_px, y0_px`` -- one row per tile whose
        centre is on tissue and whose full window fits inside the canvas.

    Raises:
        ValueError: ``stride`` is not a positive integer.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    w0, h0 = canvas_wh_px
    step = tile_um / mpp
    fov = int(round(fov_um / mpp))
    ny, nx = grid_shape(canvas_wh_px, mpp, tile_um)

    scale = tissue.shape[1] / w0

    # Subsampling the GRID (not the resulting tile list) keeps the kept tiles
    # on their exact original cell centres, so floor(cx_px / step) still
    # recovers the right cell and a strided preview lands in the same raster
    # as a full run.
    gy, gx = np.mgrid[0:ny:stride, 0:nx:stride]
    cx = (gx + 0.5) * step
    cy = (gy + 0.5) * step
    si = np.clip((cy * scale).astype(int), 0, tissue.shape[0] - 1)
    sj = np.clip((cx * scale).astype(int), 0, tissue.shape[1] - 1)
    on = tissue[si, sj]

    x0 = (cx - fov / 2).astype(int)
    y0 = (cy - fov / 2).astype(int)
    ok = on & (x0 >= 0) & (y0 >= 0) & (x0 + fov <= w0) & (y0 + fov <= h0)

    return pd.DataFrame(
        {
            "cx_px": cx[ok].astype(int),
            "cy_px": cy[ok].astype(int),
            "x0_px": x0[ok],
            "y0_px": y0[ok],
        }
    )


# --------------------------------------------------------------------------
# embedding + classification
# --------------------------------------------------------------------------


def pick_device() -> str:
    """``mps`` > ``cuda`` > ``cpu``, matching the original script's preference."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_encoder(device: str | None = None):
    """Load the frozen UNI2-h feature extractor.

    Returns:
        ``(model, device, dtype)``.

    Raises:
        RuntimeError: the gated weights could not be fetched -- request access
            at <https://huggingface.co/MahmoodLab/UNI2-h> and ``hf auth login``.
    """
    import timm
    import torch

    device = device or pick_device()
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    kw = dict(
        img_size=_EMBED_PX,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=_EMBED_DIM,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    try:
        model = (
            timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kw)
            .eval()
            .to(device, dtype)
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load MahmoodLab/UNI2-h: {exc!r}. If that is an "
            f"authorisation error: it is a GATED model -- request access at "
            f"https://huggingface.co/MahmoodLab/UNI2-h, then run `hf auth "
            f"login` with a read token (huggingface_hub 1.x renamed the CLI "
            f"-- it is `hf auth login`, not `huggingface-cli login`). Offline "
            f"(HF_HUB_OFFLINE=1), the weights must already be cached, and an "
            f"SSL_CERT_FILE pointing at a missing file still breaks the load."
        ) from exc
    return model, device, dtype


def tile_fingerprint(tiles: pd.DataFrame, mpp: float, fov_um: float) -> str:
    """Stable hash of the exact tile set an embedding cache belongs to.

    Guards against the failure that makes caching worse than useless: silently
    reusing embeddings computed for a DIFFERENT tile set. Any change to the
    tile coordinates, the resolution or the window size produces a different
    fingerprint and invalidates the cache.
    """
    import hashlib

    h = hashlib.sha256()
    for column in ("cx_px", "cy_px", "x0_px", "y0_px"):
        h.update(np.ascontiguousarray(tiles[column].to_numpy(np.int64)).tobytes())
    h.update(f"{mpp:.10g}|{fov_um:.10g}|{_EMBED_DIM}".encode())
    return h.hexdigest()


def load_embedding_cache(
    cache_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Load an embedding cache written by :func:`embed_tiles`.

    Returns:
        ``(emb, done, fingerprint)`` -- the ``(N, 1536)`` float16 embeddings,
        an ``(N,)`` bool mask of which rows are computed, and the tile-set
        fingerprint. ``None`` if the file is absent or unreadable.

    The embeddings are the only expensive part of the pipeline (UNI2-h at a
    few tiles/s); the classifier on top runs in milliseconds. Load a cache to
    re-score a slide with a different quantile, or with a retrained
    classifier, without paying for the encoder again::

        emb, done, _ = load_embedding_cache("out/embeddings.npz")
        p, cols = classify_embeddings(emb[done], load_model(), quantile=0.9)
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path) as z:
            return z["emb"], z["done"], str(z["fingerprint"])
    except (OSError, KeyError, ValueError, EOFError):
        return None


def _write_embedding_cache(
    cache_path: Path, emb: np.ndarray, done: np.ndarray, fingerprint: str
) -> None:
    """Atomically persist the cache: write a temp file, then rename over.

    A rename is atomic on POSIX, so a kill part-way through a write leaves the
    previous good cache intact rather than a truncated file.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    # Write through an open handle, NOT a path: np.savez appends ".npz" to any
    # path that lacks it, which would silently put the data at
    # "<name>.npz.tmp.npz" and leave the rename below pointing at nothing.
    with open(tmp, "wb") as fh:
        np.savez(fh, emb=emb, done=done, fingerprint=np.array(fingerprint))
    tmp.replace(cache_path)


def embed_tiles(
    level0,
    tiles: pd.DataFrame,
    *,
    mpp: float,
    fov_um: float = FOV_UM,
    encoder=None,
    device: str | None = None,
    batch: int = BATCH,
    block: int = BLOCK,
    verbose: bool = True,
    cache_path: str | Path | None = None,
    flush_every: int = 2048,
) -> np.ndarray:
    """Embed every tile window with UNI2-h.

    Reads the slide in ``block``-sized chunks and crops tiles out of each,
    so each region of the slide is decoded once rather than once per tile.

    With ``cache_path`` the embeddings are checkpointed to disk every
    ``flush_every`` tiles and reloaded on a re-run, so an interrupted section
    resumes instead of restarting. This matters at scale: a 25 000-tile
    section is over an hour, and a walltime kill would otherwise discard all
    of it.

    Args:
        level0: an array-like supporting ``[y0:y1, x0:x1]`` on the level-0
            image (e.g. the zarr array from ``TiffFile.series[0].aszarr()``).
        tiles: output of :func:`tile_grid`.
        mpp: level-0 microns per pixel.
        fov_um: window side, microns.
        encoder: preloaded ``(model, device, dtype)`` from :func:`load_encoder`;
            loaded on demand when None.
        device: device override, used only when ``encoder`` is None.
        batch: tiles per forward pass.
        block: slide read-block side, level-0 pixels.
        verbose: log throughput and ETA.
        cache_path: ``.npz`` to checkpoint embeddings to and resume from. A
            cache whose fingerprint does not match this tile set is ignored
            and overwritten, never partially reused.
        flush_every: checkpoint interval, in tiles.

    Returns:
        ``(len(tiles), 1536)`` float16 embeddings, row-aligned to ``tiles``.
    """
    import torch
    from PIL import Image

    fov = int(round(fov_um / mpp))
    h0, w0 = level0.shape[:2]

    # Row position, not the DataFrame's index label -- tiles may be sliced
    # (``--limit``) or filtered upstream, and emb is indexed positionally.
    work = tiles.reset_index(drop=True)
    fingerprint = tile_fingerprint(work, mpp, fov_um)

    emb = np.zeros((len(work), _EMBED_DIM), np.float16)
    computed = np.zeros(len(work), bool)

    cache_path = Path(cache_path) if cache_path is not None else None
    if cache_path is not None:
        cached = load_embedding_cache(cache_path)
        if cached is not None:
            cached_emb, cached_done, cached_fp = cached
            if cached_fp == fingerprint and cached_emb.shape == emb.shape:
                emb, computed = cached_emb.copy(), cached_done.copy()
                if verbose and computed.any():
                    log(
                        f"  resuming from {cache_path.name}: "
                        f"{int(computed.sum()):,}/{len(work):,} tiles cached"
                    )
            elif verbose:
                log(
                    f"  [WARN] {cache_path.name} does not match this tile set "
                    f"(different mpp, window or tile grid); recomputing."
                )

    if computed.all():
        return emb

    if encoder is None:
        encoder = load_encoder(device)
    model, dev, dtype = encoder

    work = work.assign(_bx=work.x0_px // block, _by=work.y0_px // block)

    buf: list[np.ndarray] = []
    idxs: list[int] = []
    done = int(computed.sum())
    start_done = done
    since_flush = 0
    t0 = time.time()

    def run_batch() -> None:
        nonlocal buf, idxs, done, since_flush
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).to(dev, dtype)
        with torch.inference_mode():
            emb[idxs] = model(x).float().cpu().numpy().astype(np.float16)
        computed[idxs] = True
        done += len(idxs)
        since_flush += len(idxs)
        buf, idxs = [], []

        if cache_path is not None and since_flush >= flush_every:
            _write_embedding_cache(cache_path, emb, computed, fingerprint)
            since_flush = 0

        if verbose:
            el = time.time() - t0
            rate = (done - start_done) / max(el, 1e-9)
            log(
                f"  {done:,}/{len(work):,}  {rate:.1f} tiles/s  "
                f"eta {(len(work) - done) / max(rate, 1e-9) / 60:.1f} min"
            )

    for (by, bx), grp in work.groupby(["_by", "_bx"]):
        rows = grp.index.to_numpy()
        if computed[rows].all():
            continue  # whole block already cached -- skip the slide read too
        yb, xb = by * block, bx * block
        blk = np.asarray(
            level0[yb : min(h0, yb + block + fov), xb : min(w0, xb + block + fov)]
        )
        for i, r in zip(rows, grp.itertuples()):
            if computed[i]:
                continue
            c = blk[
                int(r.y0_px) - yb : int(r.y0_px) - yb + fov,
                int(r.x0_px) - xb : int(r.x0_px) - xb + fov,
                :3,
            ]
            if c.shape[0] != fov or c.shape[1] != fov:
                continue
            im = np.asarray(
                Image.fromarray(c).resize((_EMBED_PX, _EMBED_PX), Image.BILINEAR)
            )
            im = macenko(im).astype(np.float32) / 255.0
            buf.append(((im - MEAN) / STD).transpose(2, 0, 1))
            idxs.append(int(i))
            if len(buf) == batch:
                run_batch()
    run_batch()

    if cache_path is not None:
        _write_embedding_cache(cache_path, emb, computed, fingerprint)

    return emb


def classify_embeddings(
    emb: np.ndarray, model: NicheModel, *, quantile: float = 0.80
) -> tuple[np.ndarray, pd.DataFrame]:
    """Turn embeddings into per-class probabilities, an argmax, and per-slide calls.

    The per-class top-quantile calls exist because a fixed global argmax
    under-calls rare classes badly (prevalence ranged 0.4-97% across the 8
    training tumours). Thresholds are therefore chosen PER SLIDE.

    Args:
        emb: ``(N, 1536)`` embeddings.
        model: loaded :class:`NicheModel`.
        quantile: per-slide quantile for the positive call (0.80 = top 20%).

    Returns:
        ``(P, columns)`` where ``P`` is ``(N, n_classes)`` float probabilities
        in ``model.classes`` order, and ``columns`` is a DataFrame of the
        ``p_*`` / ``argmax_class`` / ``call_*`` columns.
    """
    from sklearn.preprocessing import normalize

    p = model.clf.predict_proba(normalize(emb.astype(np.float32)))

    out = pd.DataFrame(index=pd.RangeIndex(len(p)))
    for i, name in enumerate(prob_columns(model.classes)):
        out[name] = p[:, i]
    out["argmax_class"] = [model.classes[i] for i in p.argmax(1)]
    for i, name in enumerate(call_columns(model.classes)):
        thr = float(np.quantile(p[:, i], quantile))
        out[name] = (p[:, i] > thr).astype(int)

    return p, out


def predict_slide(
    slide_path: str | Path,
    *,
    mpp: float | None = None,
    model: NicheModel | str | Path | None = None,
    quantile: float = 0.80,
    mask_level: int = 4,
    limit: int = 0,
    stride: int = 1,
    device: str | None = None,
    encoder=None,
    cache_path: str | Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Predict niches for one slide.

    Args:
        slide_path: pyramidal TIFF / OME-TIFF. For a 3D reconstruction this
            should be a **registered** section from
            ``registration.warp_and_save_section``, not a raw CZI -- see
            ``docs/niches/INTEGRATION.md``.
        mpp: level-0 microns per pixel; read from the file's own metadata when
            None (see :func:`read_mpp`).
        model: a :class:`NicheModel`, a bundle path, or None for the packaged
            bundle.
        quantile: per-slide quantile for ``call_*``.
        mask_level: pyramid level used for tissue detection.
        limit: cap the number of tiles (debugging only; 0 = no cap). Takes the
            FIRST n tiles in block order, so it samples one corner of the
            section -- use ``stride`` for a representative preview.
        stride: keep every ``stride``-th grid cell in each axis; a fast
            full-extent preview at 1/stride^2 the cost.
        device: torch device override.
        encoder: preloaded encoder from :func:`load_encoder`, to amortise the
            ~2.5 GB model load across a whole section stack.
        cache_path: ``.npz`` to checkpoint embeddings to and resume from.
        verbose: log progress.

    Returns:
        ``(tiles, meta)``. ``tiles`` has ``cx_px, cy_px, x0_px, y0_px``, one
        ``p_<class>`` per class, ``argmax_class``, and one ``call_<class>`` per
        class. ``meta`` carries ``mpp``, ``canvas_wh``, ``classes``,
        ``tile_um``, ``fov_um``, ``n_tiles``, ``tissue_frac``, ``mask_level``.

    Raises:
        ValueError: no tiles landed on tissue -- check ``mpp`` and the slide.
    """
    import tifffile
    import zarr

    slide_path = Path(slide_path)
    if not isinstance(model, NicheModel):
        model = load_model(model)

    if mpp is None:
        mpp = read_mpp(slide_path)
        if verbose:
            log(f"mpp {mpp:.6g} um/px (read from {slide_path.name} metadata)")
    elif verbose:
        log(f"mpp {mpp:.6g} um/px (caller-supplied, overriding file metadata)")

    with tifffile.TiffFile(str(slide_path)) as tf:
        series = tf.series[0]
        levels = series.levels
        h0, w0 = levels[0].shape[:2]
        group = zarr.open(series.aszarr(), mode="r")
        # aszarr() gives a GROUP for a pyramidal series, a bare array otherwise.
        level0 = group["0"] if hasattr(group, "array_keys") else group

        if verbose:
            log(
                f"slide {slide_path.name}  {w0}x{h0} px  "
                f"= {w0 * mpp / 1000:.1f} x {h0 * mpp / 1000:.1f} mm  "
                f"({len(levels)} pyramid levels)"
            )

        ml = min(mask_level, len(levels) - 1)
        if ml != mask_level and verbose:
            log(
                f"[WARN] mask_level {mask_level} requested but the pyramid has "
                f"only {len(levels)} level(s); using level {ml}. Tissue "
                f"detection at a fine level is slow and memory-hungry."
            )
        small = levels[ml].asarray()
        tissue = tissue_mask(small)
        if verbose:
            log(f"tissue: {100 * tissue.mean():.1f}% of frame at level {ml}")

        tiles = tile_grid(
            (w0, h0),
            mpp,
            tissue,
            tile_um=model.tile_um,
            fov_um=model.fov_um,
            stride=stride,
        )
        if limit:
            tiles = tiles.iloc[:limit]
        if verbose:
            fov = int(round(model.fov_um / mpp))
            spacing = (
                ""
                if stride == 1
                else f" (stride {stride}: every {stride * model.tile_um:.0f}um)"
            )
            log(
                f"{len(tiles):,} tiles on a {model.tile_um:.0f}um grid, "
                f"{fov}px ({model.fov_um:.0f}um) windows{spacing}"
            )
        if not len(tiles):
            raise ValueError(
                f"No tiles on tissue in {slide_path.name} -- check mpp "
                f"({mpp}) and the slide."
            )

        emb = embed_tiles(
            level0,
            tiles,
            mpp=mpp,
            fov_um=model.fov_um,
            encoder=encoder,
            device=device,
            cache_path=cache_path,
            verbose=verbose,
        )

    _, cols = classify_embeddings(emb, model, quantile=quantile)
    tiles = pd.concat(
        [tiles.reset_index(drop=True), cols.reset_index(drop=True)], axis=1
    )

    meta = {
        "slide": slide_path.name,
        "mpp": float(mpp),
        "canvas_wh": (int(w0), int(h0)),
        "classes": list(model.classes),
        "tile_um": float(model.tile_um),
        "fov_um": float(model.fov_um),
        "n_tiles": int(len(tiles)),
        "tissue_frac": float(tissue.mean()),
        "mask_level": int(ml),
        "quantile": float(quantile),
        "stride": int(stride),
    }
    return tiles, meta


# --------------------------------------------------------------------------
# CLI  (output files identical to the original predict_niches.py)
# --------------------------------------------------------------------------


def _report_grids(
    tiles: pd.DataFrame,
    classes: list[str],
    mpp: float,
    canvas: tuple[int, int],
    report_um: float = REPORT_UM,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Mean per-class probability on a coarse ``report_um`` grid, NaN off-tissue."""
    w0, h0 = canvas
    cell = report_um / mpp
    bnx, bny = int(np.ceil(w0 / cell)), int(np.ceil(h0 / cell))
    ii = np.clip((tiles.cy_px / cell).astype(int), 0, bny - 1)
    jj = np.clip((tiles.cx_px / cell).astype(int), 0, bnx - 1)
    cnt = np.zeros((bny, bnx))
    np.add.at(cnt, (ii, jj), 1)

    grids = {}
    for name, col in zip(classes, prob_columns(classes)):
        tot = np.zeros((bny, bnx))
        np.add.at(tot, (ii, jj), tiles[col].to_numpy())
        g = np.full((bny, bnx), np.nan)
        g[cnt > 0] = (tot / np.maximum(cnt, 1))[cnt > 0]
        grids[name] = g
    return grids, cnt


def _figure(
    slide_path: Path,
    tiles: pd.DataFrame,
    grids: dict[str, np.ndarray],
    classes: list[str],
    auc: dict[str, float],
    canvas: tuple[int, int],
    out_png: Path,
) -> None:
    """H&E + most-likely-niche + one probability map per class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tifffile
    from matplotlib.colors import ListedColormap

    w0, h0 = canvas
    ncol = len(classes) + 2
    fig, ax = plt.subplots(1, ncol, figsize=(6.2 * ncol, 6.6 * h0 / w0 + 1.4))

    with tifffile.TiffFile(str(slide_path)) as tf:
        levels = tf.series[0].levels
        ax[0].imshow(levels[min(5, len(levels) - 1)].asarray(), extent=[0, w0, h0, 0])
    ax[0].set_title("H&E")

    pal = {
        "epithelium": "#7b3f9d",
        "immune": "#c9553f",
        "stroma": "#3f7f9d",
        "acellular": "#b8b0a4",
    }
    stack = np.dstack([grids[c] for c in classes])
    ok = ~np.isnan(stack).all(2)
    lab = np.full(stack.shape[:2], np.nan)
    lab[ok] = np.nanargmax(stack[ok], axis=1)
    ax[1].imshow(
        lab,
        extent=[0, w0, h0, 0],
        interpolation="nearest",
        cmap=ListedColormap([pal.get(c, "#888") for c in classes]),
        vmin=0,
        vmax=len(classes) - 1,
    )
    ax[1].set_title(f"most likely niche ({REPORT_UM:.0f}um)")

    for k, c in enumerate(classes):
        im = ax[k + 2].imshow(
            grids[c],
            extent=[0, w0, h0, 0],
            cmap="magma_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax[k + 2].set_title(f"{c}  (AUC {auc.get(c, float('nan')):.2f})")
        plt.colorbar(im, ax=ax[k + 2], fraction=0.04)

    for a in ax:
        a.set_xlim(0, w0)
        a.set_ylim(h0, 0)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=105, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Predict Xenium tissue neighbourhoods (epithelium/immune/stroma/"
            "acellular) from an H&E whole-slide image."
        )
    )
    ap.add_argument("slide", help="pyramidal TIFF / OME-TIFF")
    ap.add_argument(
        "--mpp",
        type=float,
        default=None,
        help=(
            "microns per pixel at level 0. Read from the file's own metadata "
            "when omitted; pass only to override metadata you know is wrong."
        ),
    )
    ap.add_argument("--out", default="niche_output")
    ap.add_argument(
        "--model",
        default=None,
        help=f"classifier bundle (default: packaged {DEFAULT_MODEL_PATH.name})",
    )
    ap.add_argument(
        "--quantile",
        type=float,
        default=0.80,
        help="per-slide quantile for the positive call (default top 20%%)",
    )
    ap.add_argument(
        "--mask-level", type=int, default=4, help="pyramid level for tissue detection"
    )
    ap.add_argument("--limit", type=int, default=0, help="debug: cap number of tiles")
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "keep every Nth grid cell in each axis -- a full-extent preview at "
            "1/N^2 the cost (e.g. --stride 4 is ~16x faster). Unlike --limit, "
            "which samples one corner, this covers the whole section."
        ),
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help=(
            "do not checkpoint embeddings to OUT/embeddings.npz. By default "
            "they are saved every 2048 tiles, so an interrupted run resumes "
            "and the classifier can be re-run without paying for UNI2-h again."
        ),
    )
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(a.model)
    log(
        f"model: {model.scheme} | classes {model.classes} | "
        f"trained on {model.trained_on}"
    )
    if model.auc_by_class:
        log(
            "validation (leave-one-tumour-out AUC): "
            + ", ".join(f"{k} {v:.3f}" for k, v in model.auc_by_class.items())
        )

    try:
        tiles, meta = predict_slide(
            a.slide,
            mpp=a.mpp,
            model=model,
            quantile=a.quantile,
            mask_level=a.mask_level,
            limit=a.limit,
            stride=a.stride,
            cache_path=None if a.no_cache else out / "embeddings.npz",
        )
    except ValueError as exc:
        log(f"[ERROR] {exc}")
        return 1

    tiles.to_csv(out / "tiles_niches.csv", index=False)

    grids, cnt = _report_grids(tiles, model.classes, meta["mpp"], meta["canvas_wh"])
    np.savez_compressed(
        out / f"niche_maps_{int(REPORT_UM)}um.npz",
        **{c.replace(" ", "_"): g for c, g in grids.items()},
        tile_count=cnt,
    )

    _figure(
        Path(a.slide),
        tiles,
        grids,
        model.classes,
        model.auc_by_class,
        meta["canvas_wh"],
        out / "niche_maps.png",
    )

    summary = dict(
        slide=meta["slide"],
        mpp=meta["mpp"],
        n_tiles=meta["n_tiles"],
        tissue_mm2=float(meta["n_tiles"] * meta["tile_um"] ** 2 / 1e6),
        blocks_250um=int(cnt.astype(bool).sum()),
        mean_prob={
            c: float(tiles[col].mean())
            for c, col in zip(model.classes, prob_columns(model.classes))
        },
        argmax_share={
            c: float((tiles["argmax_class"].to_numpy() == c).mean())
            for c in model.classes
        },
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log("\n" + json.dumps(summary, indent=2))
    log(
        f"\nwrote {out}/: tiles_niches.csv, niche_maps_{int(REPORT_UM)}um.npz, "
        f"niche_maps.png, summary.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
