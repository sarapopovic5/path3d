"""Visualization helpers for path3d outputs.

2D QC overlays for visually inspecting segmentation label maps against their
source H&E tiles. Always saves to file rather than displaying interactively --
safe for headless use on a remote/HPC cluster with no display.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.color import label2rgb
from skimage.util import img_as_ubyte

from path3d.config import TISSUE_CLASSES

# Fixed per-class colors, indexed by class number.
_CLASS_COLORS = (
    (0.10, 0.85, 0.85),  # 0 cyan-space
    (0.90, 0.10, 0.10),  # 1 red-epithelium
    (0.10, 0.60, 0.20),  # 2 green-stroma
    (0.10, 0.30, 0.90),  # 3 blue-rbc
    (0.95, 0.80, 0.10),  # 4 yellow-intraluminal secretion
)


def overlay_label_map(
    label_map: np.ndarray,
    tile_rgb: np.ndarray,
    tissue_type: str,
    output_path: str | Path,
    *,
    alpha: float = 0.4,
) -> np.ndarray:
    """Alpha-blend a segmentation label map over its source tile and save to disk.

    Each tissue class is rendered in a fixed color from ``_CLASS_COLORS``,
    including class 0 (space, cyan). Writes a PNG instead of displaying
    interactively.

    Args:
        label_map: (H, W) int32 array, values in [0, num_classes) as produced
            by segmentation.predict_section.
        tile_rgb: (H, W, 3) uint8 RGB source tile, same (H, W) as label_map.
        tissue_type: key into config.TISSUE_CLASSES (e.g. "HGSC").
        output_path: PNG file path to write the overlay to.
        alpha: opacity of the label overlay, in [0, 1].

    Returns:
        (H, W, 3) uint8 RGB overlay image -- the same array written to disk.

    Raises:
        ValueError: if tissue_type is unknown, label_map/tile_rgb shapes
            don't match, or there aren't enough colors defined for
            tissue_type's number of classes.
    """
    if tissue_type not in TISSUE_CLASSES:
        raise ValueError(f"Unknown tissue_type: {tissue_type}")
    if label_map.shape != tile_rgb.shape[:2]:
        raise ValueError(
            f"label_map shape {label_map.shape} does not match "
            f"tile_rgb shape {tile_rgb.shape[:2]}"
        )

    num_classes = len(TISSUE_CLASSES[tissue_type])
    if num_classes > len(_CLASS_COLORS):
        raise ValueError(
            f"{tissue_type} has {num_classes} classes but only "
            f"{len(_CLASS_COLORS)} colors are defined in _CLASS_COLORS"
        )

    overlay = label2rgb(
        label_map,
        image=tile_rgb,
        colors=_CLASS_COLORS[:num_classes],
        alpha=alpha,
        bg_label=-1,
        image_alpha=1,
    )
    overlay_uint8 = img_as_ubyte(overlay)

    from PIL import Image
    Image.fromarray(overlay_uint8).save(str(output_path))

    return overlay_uint8
