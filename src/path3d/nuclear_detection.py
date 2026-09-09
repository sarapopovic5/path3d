"""Nuclear detection from H&E tiles via Cellpose-SAM

Produces per-section nuclear instance data (centroids + morphometrics) for downstream quantification

operates on original slide tiles read directly via `slide_io` — never on
registered or segmented output.
detection runs first per-section; registration transforms are applied
afterward to the resulting coordinates (via registration.py's warp_xy
point-transform), not to the source image

Coordinate conventions: 0-based, (row, col) = (y, x). Centroids returned
by `detect_nuclei_section` are in level-0 pixel space. Morphometric
features (area, feret diameter) are reported in physical microns


"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

import path3d.config as cfg

if TYPE_CHECKING:
    from path3d.slide_io import SlideReader


_FEATURE_COLUMNS = (
    "centroid_y",
    "centroid_x",
    "area_um2",
    "feret_diameter_um",
    "eccentricity",
    "orientation",
    "solidity",
)


def _load_cellpose_model(gpu: bool) -> Any:
    """Instantiate a Cellpose-SAM model

    Args:
        gpu: Run inference on GPU (CUDA or MPS) if available.

    Returns:
        A `cellpose.models.CellposeModel` configured with the `cpsam_v2` weights.

    Raises:
        ImportError: cellpose is not installed
    """
    try:
        from cellpose import models
    except ImportError as exc:
        raise ImportError(
            "cellpose>=4 is required for nuclear detection. "
            "Install it with:  pip install cellpose"
        ) from exc

    return models.CellposeModel(gpu=gpu, pretrained_model="cpsam_v2")


def detect_nuclei_tile(
    tile_rgb: np.ndarray,
    *,
    model: Any | None = None,
    gpu: bool = True,
) -> np.ndarray:
    """Detect nuclei in a single tile using Cellpose-SAM

    Args:
        tile_rgb: (H, W, 3) uint8 RGB tile.
        model: A pre-loaded `CellposeModel` If None, a model is loaded for this call alone.
        gpu: Passed to `_load_cellpose_model` when `model` is None; ignored
            if `model` is provided.

    Returns:
        (H, W) int32 instance label map. 0 = background, 1..N = nucleus IDs.

    Raises:
        ImportError: cellpose is not installed and no model was supplied
        ValueError: tile_rgb is not an (H, W, 3) array
    """
    if tile_rgb.ndim != 3 or tile_rgb.shape[2] != 3:
        raise ValueError(
            f"tile_rgb must be (H, W, 3), got shape {tile_rgb.shape}"
        )

    if model is None:
        model = _load_cellpose_model(gpu)

    masks, _, _ = model.eval(tile_rgb)  # type: ignore
    return masks.astype(np.int32)


def _extract_instance_features_with_labels(
    label_map: np.ndarray, *, mpp: float
) -> tuple[pd.DataFrame, list[int]]:
    """Like `_extract_instance_features`, but also returns each row's
    original Cellpose label id — used internally to build the global-ID
    remap table for persisted instance masks 

    Args:
        label_map: (H, W) int instance label map. 0 = background
        mpp: of `label_map`.

    Returns:
        (feats, local_labels) where `feats` matches `_extract_instance_features`'s
        output and `local_labels[i]` is the label id of row `i`. Ids may be
        non-contiguous (spurious masks are filtered), so don't assume
        `local_labels[i] == i + 1`.
    """
    from skimage.measure import regionprops

    regions = regionprops(label_map)
    local_labels = [r.label for r in regions]
    feats = _extract_instance_features(label_map, mpp=mpp)
    return feats, local_labels


def _extract_instance_features(label_map: np.ndarray, *, mpp: float) -> pd.DataFrame:
    """Compute per-instance morphometric features from an instance label map

    Args:
        label_map: (H, W) int instance label map. 0 = background.
        mpp: microns per pixel of `label_map`, used to convert pixel-based
            measurements to physical units.

    Returns:
        DataFrame, one row per instance, columns:
            centroid_y, centroid_x 
            area_um2               
            feret_diameter_um    
            eccentricity           
            orientation             
            solidity                
        Empty DataFrame with the same columns if `label_map` has no instances.
    """
    from skimage.measure import regionprops

    regions = regionprops(label_map)
    if not regions:
        return pd.DataFrame(columns=list(_FEATURE_COLUMNS))

    rows = []
    for r in regions:
        centroid_y, centroid_x = r.centroid
        rows.append(
            {
                "centroid_y": centroid_y,
                "centroid_x": centroid_x,
                "area_um2": r.area * mpp**2,
                "feret_diameter_um": r.feret_diameter_max * mpp,
                "eccentricity": r.eccentricity,
                "orientation": r.orientation,
                "solidity": r.solidity,
            }
        )
    return pd.DataFrame(rows, columns=list(_FEATURE_COLUMNS))


def detect_nuclei_section(
    slide: "SlideReader | str | Path",
    *,
    seg_mpp: float = cfg.SEG_MPP,
    tile_size: int = 1024,
    overlap: int = 256,
    gpu: bool = True,
    output_zarr: str | Path | None = None,
) -> pd.DataFrame | dict[str, Any]:
    """Detect nuclei across a whole section by tiling a slide.

    Each nucleus is counted once, via a non-overlapping core window per
    tile (centroid must fall inside the tile's core). If `output_zarr` is
    set, kept instances are also persisted as a chunked uint32 zarr label
    mask with globally-unique IDs (counter resets per call).

    Args:
        slide: An open SlideReader, or a path (opened/closed internally).
        seg_mpp: Target resolution in microns/pixel.
        tile_size: Tile edge length in level pixels.
        overlap: Extra context read per tile side; doesn't affect counting.
        gpu: Run Cellpose-SAM on GPU if available.
        output_zarr: Optional path to also persist the instance label mask.

    Returns:
        DataFrame of detections (centroid_y, centroid_x, area_um2,
        feret_diameter_um, eccentricity, orientation, solidity), or if
        `output_zarr` is set, `{"features": <that DataFrame>, "mask_path":
        Path(output_zarr)}`.

    Raises:
        FileNotFoundError: `slide` path doesn't exist
        ValueError: slide is missing MPP calibration, or `output_zarr`'s
            parent directory could not be created.
    """
    if isinstance(slide, (str, Path)):
        from path3d.slide_io import open_slide

        with open_slide(slide) as opened:
            return _detect_nuclei_section(
                opened,
                seg_mpp=seg_mpp,
                tile_size=tile_size,
                overlap=overlap,
                gpu=gpu,
                output_zarr=output_zarr,
            )
    return _detect_nuclei_section(
        slide,
        seg_mpp=seg_mpp,
        tile_size=tile_size,
        overlap=overlap,
        gpu=gpu,
        output_zarr=output_zarr,
    )


def _detect_nuclei_section(
    slide: "SlideReader",
    *,
    seg_mpp: float,
    tile_size: int,
    overlap: int,
    gpu: bool,
    output_zarr: str | Path | None = None,
) -> pd.DataFrame | dict[str, Any]:
    level = slide.best_level_for_mpp(seg_mpp)
    ds = slide.level_downsamples[level]
    level_mpp = slide.get_mpp() * ds
    level_w, level_h = slide.level_dimensions[level]

    model = _load_cellpose_model(gpu)

    output_canvas = None
    next_global_id = [0]
    if output_zarr is not None:
        try:
            import zarr
        except ImportError as exc:
            raise ImportError(
                "zarr is required for persisted instance masks. "
                "Install it with:  pip install \"zarr>=2.16,<3\""
            ) from exc

        mask_path = Path(output_zarr)
        try:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(
                f"Could not create parent directory for output_zarr: {output_zarr}"
            ) from exc

        output_canvas = zarr.open(
            str(mask_path),
            mode="w",
            shape=(level_h, level_w),
            chunks=(tile_size, tile_size),
            dtype="uint32",
            fill_value=0,
        )

    all_rows = []
    for (col_idx, row_idx), _actual_size, tile_rgb in slide.iter_tiles(
        level, tile_size=tile_size, overlap=overlap
    ):
        x0 = max(0, col_idx * tile_size - overlap)
        y0 = max(0, row_idx * tile_size - overlap)

        core_x_lo = col_idx * tile_size
        core_y_lo = row_idx * tile_size
        core_x_hi = min(level_w, core_x_lo + tile_size)
        core_y_hi = min(level_h, core_y_lo + tile_size)

        label_map = detect_nuclei_tile(tile_rgb, model=model)
        if output_canvas is not None:
            feats, local_labels = _extract_instance_features_with_labels(
                label_map, mpp=level_mpp
            )
        else:
            feats = _extract_instance_features(label_map, mpp=level_mpp)
            local_labels = None
        if feats.empty:
            continue

        global_y = y0 + feats["centroid_y"]
        global_x = x0 + feats["centroid_x"]
        in_core = (
            (global_x >= core_x_lo)
            & (global_x < core_x_hi)
            & (global_y >= core_y_lo)
            & (global_y < core_y_hi)
        )

        if output_canvas is not None:
            assert local_labels is not None  # set above whenever output_canvas is set
            remap = np.zeros(int(label_map.max()) + 1, dtype=np.uint32)
            for lbl, keep in zip(local_labels, in_core):
                if keep:
                    next_global_id[0] += 1
                    remap[lbl] = next_global_id[0]
            remapped = remap[label_map]

            local_core_y0 = core_y_lo - y0
            local_core_y1 = core_y_hi - y0
            local_core_x0 = core_x_lo - x0
            local_core_x1 = core_x_hi - x0
            output_canvas[core_y_lo:core_y_hi, core_x_lo:core_x_hi] = remapped[
                local_core_y0:local_core_y1, local_core_x0:local_core_x1
            ]

        kept = feats.loc[in_core].copy()
        if kept.empty:
            continue

        kept["centroid_y"] = (y0 + kept["centroid_y"]) * ds
        kept["centroid_x"] = (x0 + kept["centroid_x"]) * ds
        all_rows.append(kept)

    if not all_rows:
        features = pd.DataFrame(columns=list(_FEATURE_COLUMNS))
    else:
        features = pd.concat(all_rows, ignore_index=True)

    if output_zarr is not None:
        return {"features": features, "mask_path": Path(output_zarr)}
    return features


def stereological_correct(
    masks_or_diameters: pd.DataFrame | list[float] | np.ndarray,
    section_thickness_um: float = 4.0,
    sampling_interval: int = 1,
) -> float:
    """Convert 2D nuclear counts to a 3D-corrected estimate. per-instance correction (uses each nucleus's own diameter)

    Args:
        masks_or_diameters: DataFrame with `feret_diameter_um` (e.g. from
            `detect_nuclei_section`), or a 1-D array-like of diameters (µm).
        section_thickness_um: True physical section thickness T (µm).
        sampling_interval: Physical sections cut per imaged section. Must be >= 1.

    Returns:
        3D-corrected nuclear count estimate

    Raises:
        ValueError: input is empty, `feret_diameter_um` is missing, or
            `sampling_interval < 1`.
    """
    if sampling_interval < 1:
        raise ValueError(
            "sampling_interval must be >= 1 (1 = every section imaged), got "
            f"{sampling_interval}"
        )

    if isinstance(masks_or_diameters, pd.DataFrame):
        if "feret_diameter_um" not in masks_or_diameters.columns:
            raise ValueError(
                "DataFrame input must have a 'feret_diameter_um' column, got "
                f"columns: {list(masks_or_diameters.columns)}"
            )
        diameters = masks_or_diameters["feret_diameter_um"].to_numpy(dtype=float)
    else:
        diameters = np.asarray(masks_or_diameters, dtype=float)

    if diameters.size == 0:
        raise ValueError("masks_or_diameters must be non-empty")

    T = section_thickness_um
    P = T * sampling_interval
    return float(np.sum(P / (T + diameters)))
