"""Quantify a niche volume from continuous probabilities, not the argmax.

``quantification.compute_volumetrics`` reads ``labels["tissue_labels"]``, which
is an argmax. That systematically inflates whichever class dominates a slide
and suppresses the rest -- the bias the model card warns about:

    A fixed global ``argmax`` under-calls rare classes even when ranking is
    excellent: on our 4.5%-epithelium tumour the epithelium AUC was 0.935
    while its F1 was 0.23.

Measured on one real HGSC section (24 598 tiles, 25.19 mm2), argmax against
the expected volume from the same probabilities:

===========  ===========  ==========  =======
class        argmax mm2   soft mm2    change
===========  ===========  ==========  =======
acellular    17.39        15.56       -11%
epithelium    4.68         5.81       +24%
immune        0.01         0.15       x16
stroma        3.11         3.67       +18%
===========  ===========  ==========  =======

:func:`soft_volumetrics` is the unbiased estimator: a voxel with
``p_epithelium = 0.6`` contributes 0.6 of a voxel of epithelium, not 1 or 0.
Summed over a compartment it is an unbiased estimate of its volume under the
model's own calibration, whereas argmax discards the magnitude entirely.

For a binary compartment (connected components, surface area, shape), use
:func:`probability_mask` with a **per-section** quantile rather than a global
cutoff -- prevalence varies enormously between sections, so a fixed threshold
does not transfer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from spatialdata import SpatialData

DEFAULT_PROBS_KEY = "niche_probabilities"
DEFAULT_LABELS_KEY = "tissue_labels"


def _probs_array(
    sdata: "SpatialData", probs_key: str = DEFAULT_PROBS_KEY
) -> tuple[np.ndarray, list[str]]:
    """``(c, z, y, x)`` probabilities plus the class names from ``coords["c"]``.

    Raises:
        KeyError: no such image element.
    """
    if probs_key not in sdata.images:
        raise KeyError(
            f"'{probs_key}' not found in sdata.images (images: "
            f"{sorted(sdata.images)}). build_niche_volume writes it; a volume "
            f"from volume.build_volume has labels only."
        )
    element = sdata.images[probs_key]
    return np.asarray(element.data), [str(c) for c in element.coords["c"].values]


def voxel_um_zyx(
    sdata: "SpatialData",
    *,
    element_key: str = DEFAULT_PROBS_KEY,
    coord_system: str = "microns_3d",
) -> tuple[float, float, float]:
    """Voxel size in microns, read from the element's own affine.

    Never assumes isotropy -- ``build_niche_volume`` keeps z at the physical
    section spacing and the in-plane axes at the niche grid, so the three
    differ by design.
    """
    from spatialdata.transformations import get_transformation

    element = (
        sdata.images[element_key]
        if element_key in sdata.images
        else sdata.labels[element_key]
    )
    matrix = get_transformation(element, coord_system).to_affine_matrix(
        input_axes=("z", "y", "x"), output_axes=("z", "y", "x")
    )
    return (float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[2, 2]))


def soft_volumetrics(
    sdata: "SpatialData",
    *,
    probs_key: str = DEFAULT_PROBS_KEY,
    labels_key: str | None = DEFAULT_LABELS_KEY,
    coord_system: str = "microns_3d",
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Expected volume per niche class, from continuous probabilities.

    A voxel contributes ``p`` of itself to each class rather than all-or-
    nothing, so the estimate uses the model's calibration instead of throwing
    it away. When ``labels_key`` is given the argmax volume is reported
    alongside, so the bias is visible rather than implicit.

    Args:
        sdata: container from :func:`path3d.niches.build_niche_volume`.
        probs_key: image element holding ``(c, z, y, x)`` probabilities.
        labels_key: label element to compare against; ``None`` to skip.
        coord_system: coordinate system to read the voxel size from.
        csv_path: optional path to also write the table.

    Returns:
        One row per class:

        ``class_name``, ``mean_prob`` (over tissue voxels),
        ``expected_volume_um3`` / ``_mm3``, ``volume_fraction`` (soft), and --
        when ``labels_key`` is given -- ``argmax_volume_mm3``,
        ``argmax_volume_fraction`` and ``soft_over_argmax``.

    Raises:
        KeyError: ``probs_key`` is not present.
    """
    probs, classes = _probs_array(sdata, probs_key)
    vz, vy, vx = voxel_um_zyx(
        sdata, element_key=probs_key, coord_system=coord_system
    )
    voxel_volume_um3 = vz * vy * vx

    tissue = np.isfinite(probs).any(axis=0)
    n_tissue = int(tissue.sum())

    argmax_counts: dict[str, int] = {}
    if labels_key is not None and labels_key in sdata.labels:
        from path3d.config import NICHE_LABEL_INDEX

        labels = np.asarray(sdata.labels[labels_key].data)
        for name in classes:
            index = NICHE_LABEL_INDEX.get(name)
            if index is not None:
                argmax_counts[name] = int((labels == index).sum())

    rows = []
    for k, name in enumerate(classes):
        channel = probs[k]
        finite = np.isfinite(channel)
        total_p = float(np.nansum(np.where(finite, channel, 0.0)))
        mean_p = total_p / n_tissue if n_tissue else float("nan")

        row = {
            "class_name": name,
            "mean_prob": mean_p,
            "expected_volume_um3": total_p * voxel_volume_um3,
            "expected_volume_mm3": total_p * voxel_volume_um3 / 1e9,
            "volume_fraction": mean_p,
            "n_tissue_voxels": n_tissue,
        }
        if name in argmax_counts:
            argmax_v = argmax_counts[name] * voxel_volume_um3
            row["argmax_volume_mm3"] = argmax_v / 1e9
            row["argmax_volume_fraction"] = (
                argmax_counts[name] / n_tissue if n_tissue else float("nan")
            )
            row["soft_over_argmax"] = (
                (total_p * voxel_volume_um3) / argmax_v
                if argmax_counts[name]
                else float("inf")
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    if csv_path is not None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    return df


def probability_mask(
    sdata: "SpatialData",
    class_name: str,
    *,
    quantile: float = 0.80,
    per_section: bool = True,
    probs_key: str = DEFAULT_PROBS_KEY,
) -> np.ndarray:
    """Binary compartment mask for one class, thresholded by quantile.

    **Per-section by default.** Prevalence varied 4.5-98.4% across the eight
    training tumours, so a single global cutoff does not transfer between
    slides -- and within one stack, sections differ in how much of each
    compartment they contain. Thresholding each z-slice at its own quantile is
    the 3D form of the model card's per-slide rule.

    Off-tissue voxels are never included regardless of threshold.

    The comparison is strictly ``>``, matching ``predict_niches.py``'s own
    ``P[:, i] > thr_c``. A consequence worth knowing: a perfectly uniform (or
    heavily tied) probability field has no "top 20%", so the mask comes back
    empty rather than containing an arbitrary subset of the ties.

    Args:
        sdata: container from :func:`path3d.niches.build_niche_volume`.
        class_name: one of the volume's class names.
        quantile: keep voxels above this quantile of the on-tissue
            probabilities. ``0.80`` keeps the top 20%.
        per_section: threshold each z-slice independently (recommended).
            ``False`` uses one threshold over the whole volume.
        probs_key: image element holding the probabilities.

    Returns:
        ``(z, y, x)`` boolean mask.

    Raises:
        KeyError: ``class_name`` is not one of the volume's classes.
        ValueError: ``quantile`` is not in ``[0, 1)``.
    """
    if not 0.0 <= quantile < 1.0:
        raise ValueError(f"quantile must be in [0, 1), got {quantile}")

    probs, classes = _probs_array(sdata, probs_key)
    if class_name not in classes:
        raise KeyError(
            f"'{class_name}' is not a class of this volume (have: {classes})."
        )
    channel = probs[classes.index(class_name)]

    mask = np.zeros(channel.shape, bool)
    if per_section:
        for z in range(channel.shape[0]):
            plane = channel[z]
            finite = np.isfinite(plane)
            if not finite.any():
                continue
            threshold = float(np.quantile(plane[finite], quantile))
            mask[z] = finite & (plane > threshold)
    else:
        finite = np.isfinite(channel)
        if finite.any():
            threshold = float(np.quantile(channel[finite], quantile))
            mask = finite & (channel > threshold)
    return mask


def add_probability_labels(
    sdata: "SpatialData",
    class_name: str,
    *,
    quantile: float = 0.80,
    per_section: bool = True,
    probs_key: str = DEFAULT_PROBS_KEY,
    labels_key: str | None = None,
    coord_system: str = "microns_3d",
) -> str:
    """Add a quantile-thresholded compartment as a ``Labels3DModel`` element.

    Bridges the continuous volume back to the tools that need integer labels
    (``skimage.measure.label``, ``regionprops``, ``view_volume``) without
    routing through the argmax. The element is ``1`` inside the compartment
    and ``0`` outside, carrying the same affine as the probability volume.

    Args:
        sdata: container to add to, modified in place.
        class_name: class to threshold.
        quantile, per_section: see :func:`probability_mask`.
        probs_key: image element holding the probabilities.
        labels_key: element name to write; defaults to ``"<class_name>_call"``.
        coord_system: coordinate system for the transform.

    Returns:
        The element name written.
    """
    from spatialdata.models import Labels3DModel
    from spatialdata.transformations import get_transformation

    mask = probability_mask(
        sdata,
        class_name,
        quantile=quantile,
        per_section=per_section,
        probs_key=probs_key,
    )
    transform = get_transformation(sdata.images[probs_key], coord_system)
    element = Labels3DModel.parse(
        mask.astype(np.uint8),
        dims=("z", "y", "x"),
        transformations={coord_system: transform},
    )
    element = element.chunk({"z": 1, "y": 512, "x": 512})

    key = labels_key or f"{class_name}_call"
    sdata.labels[key] = element
    return key


def per_section_profile(
    sdata: "SpatialData",
    *,
    probs_key: str = DEFAULT_PROBS_KEY,
    coord_system: str = "microns_3d",
) -> pd.DataFrame:
    """Expected area per class for each z-section, for trends through the block.

    Useful as a registration and QC check as much as a result: a compartment
    that jumps discontinuously between adjacent sections usually means a
    registration failure or a section artefact, not biology.

    Returns:
        One row per (section, class): ``section_index``, ``z_um``,
        ``class_name``, ``mean_prob``, ``expected_area_mm2``,
        ``n_tissue_voxels``.
    """
    probs, classes = _probs_array(sdata, probs_key)
    vz, vy, vx = voxel_um_zyx(
        sdata, element_key=probs_key, coord_system=coord_system
    )
    pixel_area_um2 = vy * vx

    rows = []
    for z in range(probs.shape[1]):
        plane = probs[:, z]
        tissue = np.isfinite(plane).any(axis=0)
        n_tissue = int(tissue.sum())
        for k, name in enumerate(classes):
            channel = plane[k]
            total_p = float(np.nansum(np.where(np.isfinite(channel), channel, 0.0)))
            rows.append(
                {
                    "section_index": z,
                    "z_um": z * vz,
                    "class_name": name,
                    "mean_prob": total_p / n_tissue if n_tissue else float("nan"),
                    "expected_area_mm2": total_p * pixel_area_um2 / 1e6,
                    "n_tissue_voxels": n_tissue,
                }
            )
    return pd.DataFrame(rows)
