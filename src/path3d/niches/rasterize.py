"""Turn a per-tile niche table into a per-section raster on the shared canvas.

``predict.predict_slide`` returns one row per 32 um tile, keyed by level-0
pixel centre. ``volume.build_niche_volume`` needs a dense ``(ny, nx)`` grid per
section, and every section's grid must be **exactly the same shape** -- it
stacks them with ``np.stack``.

That works because all sections warped by
``registration.warp_and_save_section`` from one VALIS run land on the same
aligned canvas. So the grid is derived from the canvas dimensions via
:func:`predict.grid_shape`, never from the tiles that happen to be on tissue.
Two sections with different amounts of tissue still produce identically-shaped
grids; the difference shows up as NaN, not as a different array size.

Probabilities are kept, not thresholded. The model card is explicit about why::

    Use the continuous ``p_*`` columns wherever you can, and the per-slide
    calls when you need a binary mask.

A global argmax under-calls rare classes even when ranking is excellent (their
4.5%-epithelium tumour: AUC 0.935, F1 0.23), so a label volume built by
argmax inherits that bias. :func:`argmax_labels` is provided because
``view_volume`` and ``compute_volumetrics`` need integers -- not because argmax
is the better representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from path3d.config import NICHE_BACKGROUND_INDEX, NICHE_LABEL_INDEX
from path3d.niches.predict import grid_shape, prob_columns


@dataclass
class NicheSection:
    """One section's niche probabilities rasterised onto the shared canvas.

    Attributes:
        probs: ``(n_classes, ny, nx)`` float32. NaN wherever no tile was
            predicted (off-tissue, or outside the canvas margin the tile
            window needs). Channel order matches :attr:`classes`.
        classes: class names, in ``probs`` channel order.
        tile_count: ``(ny, nx)`` int32, tiles contributing to each cell --
            0 or 1 for a native-resolution grid, higher only if the caller
            rasterised at a coarser ``tile_um`` than the tiles were predicted on.
        grid_um: grid cell side in microns (in-plane voxel size downstream).
        canvas_wh: level-0 ``(width, height)`` the grid was derived from.
        mpp: level-0 microns per pixel.
    """

    probs: np.ndarray
    classes: list[str]
    tile_count: np.ndarray
    grid_um: float
    canvas_wh: tuple[int, int]
    mpp: float

    @property
    def shape(self) -> tuple[int, int]:
        """``(ny, nx)``."""
        return self.probs.shape[1:]  # type: ignore[return-value]

    @property
    def tissue_frac(self) -> float:
        """Fraction of grid cells carrying a prediction."""
        return float((self.tile_count > 0).mean())


def rasterize_tiles(
    tiles: pd.DataFrame,
    canvas_wh: tuple[int, int],
    mpp: float,
    classes: list[str],
    *,
    grid_um: float = 32.0,
) -> NicheSection:
    """Rasterise a per-tile niche table onto the canvas-derived grid.

    Args:
        tiles: table from :func:`predict.predict_slide` -- needs ``cx_px``,
            ``cy_px`` and one ``p_<class>`` column per entry in ``classes``.
        canvas_wh: level-0 ``(width, height)`` of the registered section.
        mpp: level-0 microns per pixel.
        classes: class names, setting the channel order of the output.
        grid_um: grid cell side, microns. Use the model's own ``tile_um``
            (32.0) for a native grid; a larger value aggregates, which is how
            you reproduce the model card's 250 um reporting resolution.

    Returns:
        A :class:`NicheSection`.

    Raises:
        KeyError: ``tiles`` is missing ``cx_px``/``cy_px`` or a ``p_<class>``
            column.
        ValueError: ``grid_um`` or ``mpp`` is not positive.
    """
    if mpp <= 0:
        raise ValueError(f"mpp must be > 0, got {mpp}")
    if grid_um <= 0:
        raise ValueError(f"grid_um must be > 0, got {grid_um}")

    missing = [c for c in ("cx_px", "cy_px") if c not in tiles.columns]
    cols = prob_columns(classes)
    missing += [c for c in cols if c not in tiles.columns]
    if missing:
        raise KeyError(
            f"tiles is missing required column(s): {missing}. Got "
            f"{sorted(tiles.columns)}."
        )

    ny, nx = grid_shape(canvas_wh, mpp, grid_um)
    if ny < 1 or nx < 1:
        raise ValueError(
            f"grid_um={grid_um} is larger than the whole canvas "
            f"({canvas_wh[0]}x{canvas_wh[1]} px at {mpp} um/px)."
        )

    step = grid_um / mpp
    # floor(cx_px / step) inverts predict.tile_grid's cx = (gx + 0.5) * step
    # exactly for any step >= 2 (i.e. any mpp <= grid_um/2, always true here).
    # Clipped anyway so a caller-supplied table can never index out of bounds.
    jj = np.clip(np.floor(tiles["cx_px"].to_numpy() / step).astype(np.int64), 0, nx - 1)
    ii = np.clip(np.floor(tiles["cy_px"].to_numpy() / step).astype(np.int64), 0, ny - 1)

    count = np.zeros((ny, nx), np.int32)
    np.add.at(count, (ii, jj), 1)

    probs = np.full((len(classes), ny, nx), np.nan, np.float32)
    filled = count > 0
    for k, col in enumerate(cols):
        total = np.zeros((ny, nx), np.float64)
        np.add.at(total, (ii, jj), tiles[col].to_numpy(dtype=np.float64))
        probs[k][filled] = (total[filled] / count[filled]).astype(np.float32)

    return NicheSection(
        probs=probs,
        classes=list(classes),
        tile_count=count,
        grid_um=float(grid_um),
        canvas_wh=(int(canvas_wh[0]), int(canvas_wh[1])),
        mpp=float(mpp),
    )


def rasterize_csv(
    csv_path: str | Path,
    canvas_wh: tuple[int, int],
    mpp: float,
    classes: list[str],
    *,
    grid_um: float = 32.0,
) -> NicheSection:
    """:func:`rasterize_tiles` on a ``tiles_niches.csv`` written by the CLI."""
    return rasterize_tiles(
        pd.read_csv(csv_path), canvas_wh, mpp, classes, grid_um=grid_um
    )


def smooth_probs(probs: np.ndarray, sigma_cells: float) -> np.ndarray:
    """NaN-aware in-plane Gaussian smoothing of a probability stack.

    The model card found per-tile agreement between serial sections
    "substantially weaker than per-region", and recommends reporting at 250 um.
    Smoothing the probabilities is the continuous form of that recommendation:
    it buys the same noise reduction without forcing a coarse grid, and leaves
    the in-plane resolution free to match the z spacing.

    Implemented as normalised convolution -- smooth ``prob * valid`` and
    ``valid`` separately, then divide -- so off-tissue NaNs neither leak into
    tissue nor pull edge values toward zero. Cells that are NaN on input stay
    NaN on output.

    Args:
        probs: ``(C, ny, nx)`` with NaN off-tissue.
        sigma_cells: Gaussian sigma in **grid cells**. For a sigma in microns,
            divide by ``NicheSection.grid_um``. ``<= 0`` returns a copy
            unchanged.

    Returns:
        ``(C, ny, nx)`` float32, same NaN pattern as the input.
    """
    if sigma_cells <= 0:
        return probs.copy()

    from scipy.ndimage import gaussian_filter

    valid = np.isfinite(probs[0])
    weight = gaussian_filter(valid.astype(np.float32), sigma_cells, mode="nearest")

    out = np.full_like(probs, np.nan, dtype=np.float32)
    for k in range(probs.shape[0]):
        filled = np.where(valid, np.nan_to_num(probs[k], nan=0.0), 0.0).astype(
            np.float32
        )
        smoothed = gaussian_filter(filled, sigma_cells, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k] = np.where(weight > 1e-8, smoothed / weight, np.nan)
    out[:, ~valid] = np.nan
    return out


def argmax_labels(probs: np.ndarray, classes: list[str]) -> np.ndarray:
    """Collapse a probability stack to ``config.NICHE_CLASSES`` label indices.

    Label indices come from :data:`config.NICHE_LABEL_INDEX`, keyed by class
    NAME -- deliberately not by channel position, because the classifier's
    ``classes_`` order is alphabetical (``acellular, epithelium, immune,
    stroma``) and would otherwise silently reassign every index if the bundle
    were ever retrained with a different class set.

    Off-tissue cells (all-NaN) become :data:`config.NICHE_BACKGROUND_INDEX`
    (0), which ``Labels3DModel`` and ``napari`` both treat as background.

    Args:
        probs: ``(C, ny, nx)`` with NaN off-tissue.
        classes: class names in ``probs`` channel order.

    Returns:
        ``(ny, nx)`` uint8 label map.

    Raises:
        KeyError: a class name has no entry in ``config.NICHE_LABEL_INDEX``.
    """
    unknown = [c for c in classes if c not in NICHE_LABEL_INDEX]
    if unknown:
        raise KeyError(
            f"No label index for niche class(es) {unknown}. Add them to "
            f"config.NICHE_LABEL_INDEX (known: {sorted(NICHE_LABEL_INDEX)})."
        )

    valid = np.isfinite(probs).any(axis=0)
    labels = np.full(probs.shape[1:], NICHE_BACKGROUND_INDEX, np.uint8)
    if not valid.any():
        return labels

    # nanargmax raises on all-NaN columns, so restrict to valid cells.
    winner = np.nanargmax(np.where(np.isfinite(probs), probs, -np.inf), axis=0)
    lookup = np.array([NICHE_LABEL_INDEX[c] for c in classes], np.uint8)
    labels[valid] = lookup[winner[valid]]
    return labels
