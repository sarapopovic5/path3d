"""VALIS registration backend using CPU-only classical feature matching.

Uses ORB + brute-force matcher instead of VALIS defaultDISK + LightGlue, though it imports by default
no GPU required

register() returns a dict of {section_index: valis_slide_object}. Callers use
the slide object directly via slide.warp_xy() for point warping and
slide.warp_img() for image warping.

JVM lifecycle: the JVM is started implicitly (by VALIS itself, and by the
init_jvm() call in warp_and_save_section()) and killed at most once per
process, at the very end of pipeline.run_pipeline(). Neither register() nor
warp_and_save_section() ever kills it

coordinate system: 
Each section's rigid transform becomes a SpatialData ``Affine`` mapping
pixel coords (at the registration mpp) to ``microns_3d`` (z, y, x), with
z = section_index × section_thickness_um. Elastic displacement fields go
in ``sdata.attrs["elastic_dxdy"]`` keyed by section_index.

Call ``attach_transforms(sdata, slides, mpp)`` directly, or pass ``sdata=``
to ``register()`` to do it automatically
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from skimage.transform import resize

import path3d.config as cfg
from path3d.slide_io import CziReader, open_slide
from path3d.utils import quiet_progress


def register(
    manifest_rows: list[dict],
    output_dir: str | Path,
    *,
    elastic_mpp: float = 8.0,
    do_non_rigid: bool = True,
    artifact_masks: dict[int, np.ndarray] | None = None,
    sdata=None,
    section_thickness_um: float = 12.0,
    image_key_pattern: str = "section_{:04d}",
    normalizer=None,
) -> dict[int, object]:
    """Register a section stack with VALIS using CPU-only ORB matching.

    Args:
        manifest_rows:        Manifest CSV rows as dicts with 'section_index' and
                              'path' keys. All slides must share a directory.
        output_dir:           VALIS output directory (thumbnails, logs). Created if absent.
        elastic_mpp:          Resolution for PNG export and elastic registration (µm/px).
        do_non_rigid:         Run elastic refinement after rigid pass.
        artifact_masks:       section_index → bool mask (True=artifact). Ignored when
                              do_non_rigid=False.
        sdata:                Optional SpatialData container. When provided,
                              ``attach_transforms()`` is called automatically so each
                              section's rigid transform is stored under ``microns_3d``.
        section_thickness_um: Section thickness in microns; sets z spacing in
                              ``microns_3d`` (z = section_index * thickness).
        image_key_pattern:    Format string for image keys already loaded in sdata
                              (e.g. ``"section_{:04d}"``).
        normalizer:           Optional fitted TorchMacenkoNormalizer; when given,
                              each section PNG is stain-normalised before VALIS sees it.

    Returns:
        dict mapping section_index → VALIS slide object. Use slide.warp_xy()
        and slide.warp_img() for warping — do not access slide.M directly.

    Raises:
        ImportError:  valis-wsi not installed.
        RuntimeError: VALIS completed but produced no transforms.
    """
    try:
        from valis import registration
    except ImportError as exc:
        raise ImportError(
            "valis-wsi is required. "
            "Install: pip install valis-wsi && brew install libvips openjdk"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1/3 PNG export
    img_dir = output_dir / "images"
    img_dir.mkdir(exist_ok=True)

    n = len(manifest_rows)
    cached = sum(1 for r in manifest_rows
                 if (img_dir / f"{int(r['section_index']):04d}.png").exists())
    print(f"\n[VALIS-CPU 1/3] Exporting {n} slides to PNG at {elastic_mpp} µm/px "
          f"({cached} cached, {n - cached} to export) ...")

    png_paths: list[str] = []
    filename_to_idx: dict[str, int] = {}

    for row in quiet_progress(manifest_rows, desc="PNG export"):
        sid = int(row["section_index"])
        png_path = img_dir / f"{sid:04d}.png"
        _export_section_png(row["path"], png_path, elastic_mpp)
        png_paths.append(str(png_path))
        filename_to_idx[png_path.stem] = sid
        filename_to_idx[png_path.name] = sid

    print(f"[VALIS-CPU 1/3] Done. Images in {img_dir}")

    # 2/3 VALIS registration
    from PIL import Image as _PIL
    _w, _h = _PIL.open(png_paths[0]).size
    max_dim = max(_w, _h)

    print(f"\n[VALIS-CPU 2/3] Registering {len(png_paths)} slides "
          f"(max_image_dim_px={max_dim}) using ORB + BF matcher (CPU only) ...")

    from valis import feature_detectors, feature_matcher

    matcher = feature_matcher.Matcher(
        feature_detector=feature_detectors.OrbFD(),  # type: ignore[arg-type]
        match_filter_method=feature_matcher.DEFAULT_RANSAC_NAME,
    )

    nr_cls = registration.DEFAULT_NON_RIGID_CLASS if do_non_rigid else None
    registrar = registration.Valis(
        str(img_dir),
        str(output_dir),
        img_list=png_paths,
        imgs_ordered=True,
        max_image_dim_px=max_dim,
        matcher=matcher,  # type: ignore[arg-type]
        non_rigid_registrar_cls=nr_cls,  # type: ignore
        # Disable entropy-based tissue masking to match registration_norm.py.
        # Normalized images can collapse colorfulness into <3 histogram bins
        # and crash threshold_multiotsu; disabling masking for both conditions
        # keeps the comparison apples-to-apples.
        crop_for_rigid_reg=False,
        create_masks=False,
    )

    slides: dict[int, object] = {}
    with _quiet_valis_registration_progress():
        registrar.register()
    print("\n[VALIS-CPU 2/3] Registration complete.")

    # 3/3 Extract slide objects
    print("\n[VALIS-CPU 3/3] Extracting slide objects ...")
    for slide_name, slide in registrar.slide_dict.items():
        fname = Path(slide_name).name
        section_id = filename_to_idx.get(fname)
        if section_id is None:
            continue
        slides[section_id] = slide

    print(f"[VALIS-CPU 3/3] Done — {len(slides)} slides extracted.")

    if not slides:
        raise RuntimeError(
            f"VALIS produced no transforms. Check output in {output_dir}."
        )

    if sdata is not None:
        attach_transforms(
            sdata,
            slides,
            elastic_mpp,
            image_key_pattern=image_key_pattern,
            section_thickness_um=section_thickness_um,
        )

    return slides


def warp_image(
    image: np.ndarray,
    slide,
    *,
    is_label_map: bool = False,
) -> np.ndarray:
    """Warp an image array using VALIS's built-in warping.

    Resizes ``image`` to the exact registered shape
    (``slide.processed_img_shape_rc``) before warping if it doesn't already
    match -- independent ``downsample_to_mpp()`` calls (mask vs. registration
    image) can drift a pixel or two apart, which otherwise sends VALIS into
    an unstable auto-rescale fallback that can raise ``bad extract area``
    (confirmed intermittently on real HPC data). Resize + warp both use
    nearest-neighbour for label maps, bilinear otherwise.

    Args:
        image: (H,W) or (H,W,C) uint8 array.
        slide: VALIS slide object from register().
        is_label_map: Use nearest-neighbour interpolation throughout, for
            categorical data.

    Returns:
        Warped array, same dtype as image, with the shape VALIS produces.
    """
    interp = "nearest" if is_label_map else "bicubic"
    src_dtype = image.dtype

    target_shape_rc = tuple(int(x) for x in slide.processed_img_shape_rc)
    if image.shape[:2] != target_shape_rc:
        order = 0 if is_label_map else 1
        image = resize(
            image,
            target_shape_rc + image.shape[2:],
            order=order,
            preserve_range=True,
            anti_aliasing=(order != 0),
        ).astype(src_dtype)

    warped = slide.warp_img(image, interp_method=interp)
    return warped.astype(src_dtype)


_MICRON_UNIT_NAMES = {"µm", "um", "micron", "microns", "μm"}


class _CziReaderAdapter:
    """Deliberate algorithmic divergence: bypass VALIS's own reader
    auto-detection for ``.czi`` sources, reading pixels through path3d's own
    JVM-free ``CziReader`` (aicspylibczi) instead.

    Root cause (confirmed via gdb on the crashing process):
    ``warp_and_save_section()`` used to call
    ``valis.slide_io.get_slide_reader(src_path)``, which for ``.czi``
    unconditionally tries ``BioFormatsSlideReader`` first. This dataset's
    CZIs are JPEGXR-compressed, which Bio-Formats cannot decode -- and
    instead of failing cleanly it segfaults natively. ``info proc mappings``
    resolved the SIGSEGV address to an anonymous ``rwxp`` region with NO
    backing file, a memory shape unique to a JIT code cache -- i.e. the
    crash is inside JIT-compiled Java bytecode, not in path3d, pyvips, or
    aicspylibczi. Since Bio-Formats always fails on this codec and VALIS
    always falls back to a non-JVM CZI reader afterward anyway (when it
    doesn't crash first), there is no reason to ever let it probe ``.czi``
    with Bio-Formats.

    This class exposes the minimal VALIS-reader-shaped surface that
    ``valis.slide_tools.warp_slide``, ``valis.slide_io.update_xml_for_new_img``,
    and ``_source_level_for_mpp`` depend on: ``.series``, ``.metadata`` (a
    real ``valis.slide_io.MetaData``), ``.scale_physical_size(level)``,
    ``.slide2vips(level, series, xywh)``, and ``.close()``. It is a drop-in
    substitute -- nothing downstream of ``reader = ...`` in
    ``warp_and_save_section`` needs to change.

    Memory note: unlike VALIS's native lazy pyvips readers, ``slide2vips``
    materialises the requested level into RAM because
    ``aicspylibczi.read_mosaic`` decodes to numpy -- acceptable because
    ``_source_level_for_mpp`` selects a downsampled level, not level 0 at
    full scan resolution.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._reader = CziReader(self._path)
        self.series = 0  # path3d's CziReader has no multi-series concept

        try:
            from valis import slide_io as _valis_io
        except ImportError as exc:
            raise ImportError(
                "valis-wsi is required. "
                "Install: pip install valis-wsi && brew install libvips openjdk"
            ) from exc

        mpp = self._reader.get_mpp()
        self.metadata = _valis_io.MetaData(
            name=self._path.stem, server="path3d.CziReader", series=0
        )
        self.metadata.slide_dimensions = list(self._reader.level_dimensions)
        self.metadata.is_rgb = True
        # CziReader._to_rgb_uint8 always normalises to 3-channel RGB uint8
        self.metadata.n_channels = 3
        self.metadata.pixel_physical_size_xyu = (mpp, mpp, "µm")

    def scale_physical_size(self, level: int) -> tuple[float, float]:
        """(x_mpp, y_mpp) at ``level``, scaled by that level's downsample."""
        mpp = (
            self.metadata.pixel_physical_size_xyu[0]
            * self._reader.level_downsamples[level]
        )
        return (mpp, mpp)

    def slide2vips(self, level: int = 0, series=None, xywh=None, *args, **kwargs):
        """Read ``level`` (or an ``xywh`` sub-region of it) as a pyvips.Image.

        Args:
            level: Pyramid level to read.
            series: Accepted and ignored (no multi-series concept here;
                present because ``warp_slide`` passes it as a keyword).
            xywh: Optional (x, y, w, h) sub-region at ``level``; full level
                extent when None.

        Returns:
            pyvips.Image wrapping an (H, W, 3) uint8 RGB array.
        """
        if xywh is None:
            w, h = self._reader.level_dimensions[level]
            x = y = 0
        else:
            x, y, w, h = xywh

        # location=(x, y) is in LEVEL-0 px per CziReader's read_region
        # contract; size=(w, h) is at the requested level.
        img = self._reader.read_region((x, y), level, (w, h))

        import pyvips

        return pyvips.Image.new_from_array(img)

    def close(self) -> None:
        self._reader.close()


def _source_level_for_mpp(reader: Any, target_mpp: float) -> int:
    """Pick the coarsest source pyramid level whose native mpp is <= target_mpp.

    Never returns a level coarser than ``target_mpp`` -- that would force
    ``warp_and_save_section`` to upsample synthesized pixels, the fake-detail
    bug this module exists to avoid. Raises instead of silently upsampling
    if even the finest level is too coarse.

    Args:
        reader: VALIS SlideReader (or test stub with the same surface) for
            the ORIGINAL slide file.
        target_mpp: Desired output resolution in µm/px.

    Returns:
        Index of the coarsest qualifying pyramid level (0.1% mpp tolerance).

    Raises:
        ValueError: pixel size unit isn't microns, or every level is coarser
            than ``target_mpp``.
    """
    unit = str(reader.metadata.pixel_physical_size_xyu[2]).strip().lower()
    if unit not in _MICRON_UNIT_NAMES:
        raise ValueError(
            f"Unsupported pixel physical size unit {unit!r} for "
            f"_source_level_for_mpp; expected microns."
        )

    n_levels = len(reader.metadata.slide_dimensions)
    tol_mpp = target_mpp * (1 + 1e-3)

    best_level = None
    for level in range(n_levels):
        level_mpp = reader.scale_physical_size(level)[0]
        if level_mpp <= tol_mpp:
            best_level = level  # keep scanning: prefer the coarsest qualifying level

    if best_level is None:
        level0_mpp = reader.scale_physical_size(0)[0]
        raise ValueError(
            f"Source level 0 mpp ({level0_mpp}) is coarser than target_mpp "
            f"({target_mpp}); refusing to upsample because it fabricates "
            f"detail that was never in the source image."
        )

    return best_level


def warp_and_save_section(
    slide: Any,
    src_path: str | Path,
    dst_path: str | Path,
    *,
    target_mpp: float,
    reg_mpp: float = cfg.REG_MPP,
    non_rigid: bool = True,
    crop: bool = True,
    interp_method: str = "bicubic",
    tile_wh: int = 1024,
    compression: str = "deflate",
) -> Path:
    """Warp one section from its ORIGINAL slide file to a pyramidal OME-TIFF
    at ``target_mpp``, streaming the warp tile-by-tile via pyvips.

    Reads the original slide directly rather than the single-level
    ``elastic_mpp`` PNG VALIS registered on (which would cap output
    resolution at ``elastic_mpp``), at the finest source pyramid level that's
    still >= ``target_mpp`` -- never a coarser one, which would reproduce the
    fake-upsample bug this function exists to avoid (see
    ``_source_level_for_mpp``). The warp stays a lazy ``pyvips.Image`` and is
    streamed straight to disk, so the full-resolution section is never held
    in RAM. ``.czi`` sources bypass VALIS's own reader and the JVM entirely
    via ``_CziReaderAdapter``.

    ``reg_mpp`` must equal the ``elastic_mpp`` this slide was registered with
    -- the output canvas scale (``reg_mpp / target_mpp``) is derived from it,
    so a mismatch silently misaligns the output.

    Args:
        slide: VALIS slide object for this section, from register().
        src_path: Path to the ORIGINAL slide file (not the registration PNG).
        dst_path: Output path; normalized to end in ``.ome.tiff``.
        target_mpp: Desired output resolution in µm/px.
        reg_mpp: The elastic_mpp this slide was registered with.
        non_rigid: Apply the elastic displacement field in addition to the
            rigid transform.
        crop: Crop method flag/name passed to ``slide.get_crop_method``.
        interp_method: pyvips interpolation method for the warp (e.g.
            "bicubic"; use "nearest" for label maps).
        tile_wh: Tile side length for the pyramidal OME-TIFF.
        compression: TIFF compression codec.

    Returns:
        The normalized ``dst_path`` that was written.

    Raises:
        ImportError: valis-wsi is not installed.
        ValueError: no VALIS slide reader is available for ``src_path``
            (non-``.czi`` only), or no source pyramid level is fine enough
            for ``target_mpp`` (see ``_source_level_for_mpp``).
    """
    try:
        from valis import registration as _valis_reg
        from valis import slide_io as _valis_io
        from valis import slide_tools as _valis_st
    except ImportError as exc:
        raise ImportError(
            "valis-wsi is required. "
            "Install: pip install valis-wsi && brew install libvips openjdk"
        ) from exc

    dst_path = Path(dst_path)
    if not dst_path.name.endswith(".ome.tiff"):
        stem = dst_path.name.split(".")[0]
        dst_path = dst_path.with_name(f"{stem}.ome.tiff")
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    src_path_str = str(src_path)
    is_czi = Path(src_path).suffix.lower() == ".czi"  # mirrors open_slide's routing
    if is_czi:
        # .czi sources bypass VALIS's reader selection and the JVM entirely --
        # see _CziReaderAdapter's docstring for the Bio-Formats/JPEGXR SIGSEGV
        # root cause this avoids.
        reader = _CziReaderAdapter(src_path_str)
    else:
        # init_jvm() is idempotent -- a re-entrant no-op when the JVM is
        # already alive. register() no longer kills the JVM, so this call
        # simply guarantees the JVM is up for Bio-Formats reads regardless of
        # whether register() already ran earlier in this process. Runs on
        # the non-czi path only -- running it before the branch decision
        # would defeat the .czi fix.
        _valis_reg.init_jvm()
        reader_cls = _valis_io.get_slide_reader(src_path_str)
        if reader_cls is None:
            raise ValueError(f"No VALIS slide reader available for {src_path_str}")
        reader = reader_cls(src_path_str)

    level = _source_level_for_mpp(reader, target_mpp)

    # slide.aligned_slide_shape_rc is the level-0 (== reg_mpp) aligned canvas
    # for this section; scaling it by reg_mpp/target_mpp puts every section
    # on the same scaled canvas, which volume.build_volume's np.stack requires.
    scale = reg_mpp / target_mpp
    aligned_shape_rc = np.ceil(
        np.asarray(slide.aligned_slide_shape_rc, dtype=float) * scale
    ).astype(int)

    crop_method = slide.get_crop_method(crop)
    bbox_xywh = None
    if crop_method is not False:
        bbox_xywh = slide.get_crop_xywh(
            crop=crop_method, out_shape_rc=aligned_shape_rc
        )[0]

    warped = _valis_st.warp_slide(
        src_path_str,
        M=slide.M,
        transformation_src_shape_rc=slide.processed_img_shape_rc,
        transformation_dst_shape_rc=slide.reg_img_shape_rc,
        aligned_slide_shape_rc=aligned_shape_rc,
        dxdy=(slide.bk_dxdy if non_rigid else None),
        level=level,
        series=reader.series,
        interp_method=interp_method,
        bbox_xywh=bbox_xywh,
        bg_color=None,
        reader=reader,
    )

    # vips resolution is pixels-per-millimetre. OpenSlideReader.get_mpp()
    # reads back openslide.mpp-x, which OpenSlide derives from these baseline
    # TIFF tags -- NOT from the OME-XML metadata set below -- so xres/yres
    # must be stamped before saving or predict_section cannot pick a level.
    warped = warped.copy(xres=1000.0 / target_mpp, yres=1000.0 / target_mpp) # type: ignore

    ome_xml = _valis_io.update_xml_for_new_img(
        img=warped,
        reader=reader,
        level=level,
        pixel_physical_size_xyu=(target_mpp, target_mpp, "µm"),
    ).to_xml()

    # The JVM is kept alive here and killed exactly once by run_pipeline after
    # all sections are warped.  JPype does not allow the JVM to be restarted
    # more than once per process, so killing it between sections can cause OS error
    with open(os.devnull, "w") as _devnull, contextlib.redirect_stdout(_devnull):
        _valis_io.save_ome_tiff(
            warped,
            dst_f=str(dst_path),
            ome_xml=ome_xml,
            tile_wh=tile_wh,
            compression=compression,
            pyramid=True,
        )

    return dst_path


def build_section_transforms(
    slides: dict[int, Any],
    mpp: float,
    section_thickness_um: float = 12.0,
) -> dict[int, Any]:
    """Build SpatialData Affine transforms from VALIS slide objects.

    Maps each section's pixel coords (at ``mpp``) to microns_3d (z, y, x),
    z = section_index × section_thickness_um. Requires spatialdata.
    """
    return {
        idx: _valis_rigid_to_affine3d(slide, mpp, idx * section_thickness_um)
        for idx, slide in slides.items()
    }


def attach_transforms(
    sdata: Any,
    slides: dict[int, Any],
    mpp: float,
    *,
    image_key_pattern: str = "section_{:04d}",
    section_thickness_um: float = 12.0,
    coord_system: str = "microns_3d",
) -> None:
    """Register rigid VALIS transforms in a SpatialData container

    Sets each section's transform on its image in ``sdata.images`` (must
    already exist, keyed by ``image_key_pattern``; sections with no matching
    key are skipped) from pixel coords to ``coord_system``. Elastic
    displacement fields can't be expressed as affines, so they're stored
    separately in ``sdata.attrs["elastic_dxdy"]`` keyed by section_index.

    Args:
        sdata: SpatialData container to update in-place.
        slides: dict from register().
        mpp: Registration resolution in µm/px (matches elastic_mpp passed to
            register()).
        image_key_pattern: Format string for image keys in sdata.images.
        section_thickness_um: Z spacing in microns.
        coord_system: Target coordinate system name.

    Raises:
        ImportError: spatialdata not installed
    """
    try:
        from spatialdata.transformations import set_transformation  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "spatialdata is required. Install: pip install spatialdata"
        ) from exc

    elastic_fields: dict[int, np.ndarray] = {}

    for section_idx, slide in sorted(slides.items()):
        image_key = image_key_pattern.format(section_idx)
        if image_key not in sdata.images:
            continue

        z_um = section_idx * section_thickness_um
        transform = _valis_rigid_to_affine3d(slide, mpp, z_um)
        set_transformation(sdata.images[image_key], transform, to_coordinate_system=coord_system)

        if hasattr(slide, "bk_dxdy") and slide.bk_dxdy is not None:
            elastic_fields[section_idx] = slide.bk_dxdy

    if elastic_fields:
        sdata.attrs.setdefault("elastic_dxdy", {}).update(elastic_fields)


# Internal helpers

def _valis_quiet_tqdm(iterable=None, *args, desc=None, total=None, **kwargs):
    """change VALIS tqdm's to quiet_progress
    """
    if iterable is None:
        iterable = range(total or 0)
    return quiet_progress(iterable, desc=desc, total=total)


@contextlib.contextmanager
def _quiet_valis_registration_progress():
    """silence VALIS's internal tqdm bars for the duration of registrar.register()
    """
    targets: list[tuple[Any, str]] = []

    try:
        import tqdm as _tqdm_module
        targets.append((_tqdm_module, "tqdm"))
    except (ImportError, AttributeError):
        pass

    try:
        from valis import serial_rigid as _serial_rigid
        targets.append((_serial_rigid, "tqdm"))
    except (ImportError, AttributeError):
        pass

    try:
        from valis import serial_non_rigid as _serial_non_rigid
        targets.append((_serial_non_rigid, "tqdm"))
    except (ImportError, AttributeError):
        pass

    saved: list[tuple[Any, str, Any]] = []
    for obj, attr in targets:
        try:
            saved.append((obj, attr, getattr(obj, attr)))
        except AttributeError:
            continue

    for obj, attr, _original in saved:
        setattr(obj, attr, _valis_quiet_tqdm)

    try:
        yield
    finally:
        for obj, attr, original in saved:
            setattr(obj, attr, original)


def _valis_rigid_to_affine3d(slide: Any, mpp: float, z_um: float) -> Any:
    """Fit a SpatialData Affine from VALIS warp_xy, embedding in microns_3d

    Samples 4 corner points, warps them through VALIS's full transform chain,
    then fits a 2D affine and scales to microns.

    Input axes: (y, x) slide pixel coords at ``mpp``. Output axes: (z, y, x)
    microns in ``microns_3d``, z = z_um.
    """
    from spatialdata.transformations import Affine  # type: ignore[import-untyped]

    w, h = slide.slide_dimensions_wh[0]  # px at registration mpp

    # 4 inset corner points (10% margin) in (x, y) slide pixel coords.
    pts_src = np.array([
        [0.1 * w, 0.1 * h],
        [0.9 * w, 0.1 * h],
        [0.1 * w, 0.9 * h],
        [0.9 * w, 0.9 * h],
    ])

    # warp_xy: (x,y) slide px → (x,y) registered px (both at mpp resolution)
    pts_dst = slide.warp_xy(pts_src, non_rigid=False, crop=True)

    # Fit 2D affine (x', y') = A @ (x, y, 1) by least squares.
    N = pts_src.shape[0]
    A_mat = np.zeros((2 * N, 6))
    b_vec = np.zeros(2 * N)
    for i, (src, dst) in enumerate(zip(pts_src, pts_dst)):
        xs, ys = src
        xd, yd = dst
        A_mat[2 * i,     :3] = [xs, ys, 1.0]
        A_mat[2 * i + 1, 3:] = [xs, ys, 1.0]
        b_vec[2 * i]     = xd
        b_vec[2 * i + 1] = yd

    params, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    # params: [a00, a01, a02,  a10, a11, a12]
    # x' = a00*x + a01*y + a02
    # y' = a10*x + a11*y + a12
    a00, a01, a02 = params[:3]
    a10, a11, a12 = params[3:]

    # Convert (x,y) → (y,x) axis order and scale registered px → microns.
    # Input to Affine: [y_px, x_px, 1].  Output: [z_um, y_um, x_um, 1].
    # y_um = mpp * y' = mpp*(a10*x + a11*y + a12) = mpp*a11*y_px + mpp*a10*x_px + mpp*a12
    # x_um = mpp * x' = mpp*(a00*x + a01*y + a02) = mpp*a01*y_px + mpp*a00*x_px + mpp*a02
    matrix = np.array([
        [0.,        0.,        z_um     ],
        [mpp * a11, mpp * a10, mpp * a12],  # y_um
        [mpp * a01, mpp * a00, mpp * a02],  # x_um
        [0.,        0.,        1.       ],
    ])

    return Affine(matrix, input_axes=("y", "x"), output_axes=("z", "y", "x"))


def _export_section_png(
    slide_path: str,
    out_path: Path,
    mpp: float,
) -> None:
    """Export slide to PNG at given mpp. No-op if file already exists."""
    if out_path.exists():
        return
    from PIL import Image
    from path3d.preprocessing import downsample_to_mpp
    with open_slide(slide_path) as slide:
        img = downsample_to_mpp(slide, mpp)
    Image.fromarray(img).save(str(out_path))
