"""Spatial statistics, volumetrics, and morphometrics on a tissue-label
SpatialData container.

Input: ``sdata.labels["tissue_labels"]`` -- ``(z, y, x)`` uint8 label
volume, isotropic at ``voxel_um`` microns/voxel (default 12.0). Optionally
``sdata.tables["nuclei"]`` -- an ``AnnData`` with per-nucleus morphometrics
in ``obs`` and ``[z, y, x]`` micron coordinates in ``obsm["spatial_3d"]``.

Output: results attach in place to ``sdata.tables["nuclei"]`` /
``["lesions"]`` where applicable, and are also returned as a DataFrame.
Every public function takes an opt-in ``csv_path=None`` kwarg to also
write a CSV.

Deliberate divergences: cell-type-ratio sub-features (e.g. FOXP3+/CD45+)
are dropped -- this pipeline is H&E-only, no protein-marker cell typing yet.
``surface_decorrelation``'s geodesic distance is a discrete mesh-graph
shortest path, not a dedicated heat-method geodesic-mesh library.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import path3d.config as cfg
from path3d.nuclear_detection import stereological_correct

if TYPE_CHECKING:
    from spatialdata import SpatialData


_VOLUMETRIC_COLUMNS = (
    "class_index",
    "class_name",
    "n_voxels",
    "volume_um3",
    "volume_mm3",
    "volume_fraction",
    "n_components",
    "component_sizes_um3",
    "surface_area_um2",
)


def _tissue_class_map(organ: str) -> dict[int, str]:
    """Look up the organ's tissue-class index->name map.

    Args:
        organ: Key into `config.TISSUE_CLASSES`, e.g. `"HGSC"`.

    Returns:
        `dict[int, str]` mapping class index to class name.

    Raises:
        KeyError: `organ` is not a key of `config.TISSUE_CLASSES`.
    """
    try:
        return cfg.TISSUE_CLASSES[organ]
    except KeyError as exc:
        raise KeyError(
            f"Unknown organ: {organ!r}. Available organs: "
            f"{sorted(cfg.TISSUE_CLASSES)}"
        ) from exc


def _non_space_class_indices(organ: str) -> tuple[int, ...]:
    """Return every class index in `organ` whose name is not `"space"`.

    Never hardcode a literal class-index range or set -- the valid
    non-space set is organ/config-dependent, not a universal constant.

    Args:
        organ: Key into `config.TISSUE_CLASSES`, e.g. `"HGSC"`.

    Returns:
        Sorted tuple of class indices excluding `"space"`.

    Raises:
        KeyError: `organ` is not a key of `config.TISSUE_CLASSES`.
    """
    class_map = _tissue_class_map(organ)
    return tuple(sorted(idx for idx, name in class_map.items() if name != "space"))


def _labels_array(sdata: "SpatialData") -> np.ndarray:
    """Read the `"tissue_labels"` element off `sdata` as a plain ndarray.

    Args:
        sdata: A `SpatialData` container with a tissue-label volume.

    Returns:
        `(z, y, x)` ndarray of tissue-class indices.

    Raises:
        KeyError: `sdata.labels` has no `"tissue_labels"` element.
    """
    if "tissue_labels" not in sdata.labels:
        raise KeyError(
            f"'tissue_labels' not found in sdata.labels. Available labels: "
            f"{sorted(sdata.labels)}"
        )
    return np.asarray(sdata.labels["tissue_labels"].data)


def _validate_positive(name: str, value: float) -> None:
    """Raise `ValueError` (echoing `value`) unless `value > 0`."""
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _write_csv(df: pd.DataFrame, csv_path: str | Path | None) -> None:
    """Write `df` to `csv_path` when given; no-op (and no filesystem side
    effect) when `csv_path is None` -- every public function in this
    module shares this single implementation.

    Args:
        df: The result table to export.
        csv_path: Destination path, or `None` to skip export entirely.

    Raises:
        ValueError: `csv_path`'s parent directory could not be created.
    """
    if csv_path is None:
        return
    path = Path(csv_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(
            f"Could not create parent directory for csv_path: {csv_path}"
        ) from exc
    df.to_csv(str(path), index=False)


def _surface_area_um2(mask: np.ndarray, voxel_um: float) -> float:
    """Estimate the physical surface area of a boolean 3D mask.

    Crops `mask` to its own bounding box, then pads 1 voxel of background on
    every side (`marching_cubes` needs a closed boundary to produce a valid
    watertight mesh) before measuring. Never call `marching_cubes` on the
    uncropped full volume -- for a realistic tissue-block-scale label
    volume this is a memory/compute blowup; cropping to the mask's own bbox
    is a strict, always-safe optimization.

    Args:
        mask: `(z, y, x)` boolean array, `True` where the compartment is
            present.
        voxel_um: Isotropic voxel size in microns.

    Returns:
        Surface area in square microns; `0.0` for an all-`False` mask.
    """
    if not mask.any():
        return 0.0

    from skimage.measure import marching_cubes, mesh_surface_area

    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    cropped = mask[mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]]
    padded = np.pad(cropped, pad_width=1, mode="constant", constant_values=False)

    verts, faces, _, _ = marching_cubes(
        padded.astype(np.uint8), level=0.5, spacing=(voxel_um,) * 3
    )
    return float(mesh_surface_area(verts, faces))


def compute_volumetrics(
    sdata: "SpatialData",
    *,
    organ: str = "HGSC",
    voxel_um: float = 12.0,
    connectivity: int = 1,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Per-compartment volumetrics on a tissue-label volume: voxel count,
    physical volume (um3/mm3), volume fraction, 3D connected-component
    count/sizes, and surface area, for every class in
    `config.TISSUE_CLASSES[organ]`.

    `connectivity=1` (6-connectivity) is the deliberate default -- skimage's
    own default (26-connectivity) silently merges corner-touching,
    visually-distinct compartments into one component.

    Args:
        sdata: `SpatialData` container with `labels["tissue_labels"]`.
        organ: Key into `config.TISSUE_CLASSES`. Defaults to `"HGSC"`.
        voxel_um: Isotropic voxel size, in microns. Must be `> 0`. Defaults
            to `12.0`.
        connectivity: `skimage.measure.label` connectivity, `(1, 2, 3)`.
            Defaults to `1`.
        csv_path: Optional path to also write the result as CSV.

    Returns:
        DataFrame with columns `_VOLUMETRIC_COLUMNS`, one row per class.
        Also attached in place at `sdata.tables["nuclei"].uns["volumes"]`
        when that table exists.

    Raises:
        ValueError: `voxel_um <= 0`, or `connectivity` not in `(1, 2, 3)`.
        KeyError: `organ` unknown, or no `"tissue_labels"` element.
    """
    _validate_positive("voxel_um", voxel_um)
    if connectivity not in (1, 2, 3):
        raise ValueError(f"connectivity must be one of (1, 2, 3), got {connectivity}")
    class_map = _tissue_class_map(organ)

    from skimage.measure import label, regionprops_table

    labels = _labels_array(sdata)
    total_voxels = labels.size

    rows = []
    for class_index, class_name in class_map.items():
        mask = labels == class_index
        n_voxels = int(mask.sum())
        volume_um3 = n_voxels * voxel_um**3
        volume_mm3 = volume_um3 / 1e9
        volume_fraction = n_voxels / total_voxels

        lab, n_components = label(mask, connectivity=connectivity, return_num=True)
        if n_components > 0:
            component_sizes_um3 = np.asarray(
                regionprops_table(
                    lab, properties=("area",), spacing=(voxel_um,) * 3
                )["area"]
            )
        else:
            component_sizes_um3 = np.array([], dtype=np.float64)

        surface_area_um2 = _surface_area_um2(mask, voxel_um)

        rows.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "n_voxels": n_voxels,
                "volume_um3": volume_um3,
                "volume_mm3": volume_mm3,
                "volume_fraction": volume_fraction,
                "n_components": int(n_components),
                "component_sizes_um3": component_sizes_um3,
                "surface_area_um2": surface_area_um2,
            }
        )

    df = pd.DataFrame(rows, columns=list(_VOLUMETRIC_COLUMNS))

    if "nuclei" in sdata.tables:
        sdata.tables["nuclei"].uns["volumes"] = df

    _write_csv(df, csv_path)
    return df


def _persist_element(sdata: "SpatialData", element_name: str) -> bool:
    """Safely re-persist an element that may already exist in `sdata`'s
    backing Zarr store.

    A bare `sdata.write_element(name)` on an existing element raises
    (spatialdata can't overwrite a subfolder of its own store), so this
    does `delete_element_from_disk` then `write_element` instead. The
    delete-then-write window is unprotected: an interruption mid-write
    loses the element.

    Args:
        sdata: A `SpatialData` container, in-memory or Zarr-backed.
        element_name: Name of an element in `sdata.tables` or `sdata.labels`.

    Returns:
        `True` if written to disk; `False` if `sdata.path is None`
        (in-memory container, nothing to persist).

    Raises:
        KeyError: `element_name` is in neither `sdata.tables` nor
            `sdata.labels`.
    """
    available = set(sdata.tables) | set(sdata.labels)
    if element_name not in available:
        raise KeyError(
            f"Element {element_name!r} not found in sdata.tables or "
            f"sdata.labels. Available elements: {sorted(available)}"
        )

    if sdata.path is None:
        return False

    sdata.delete_element_from_disk(element_name)
    sdata.write_element(element_name)
    return True


# ---------------------------------------------------------------------------
# Per-lesion 3D connected-component feature table.
# ---------------------------------------------------------------------------

# docs/porting_plan.md's per-lesion feature spec includes cell-type-ratio
# sub-features (e.g. FOXP3+/CD45+ ratio) borrowed from IHC/mIF pipelines.
# This pipeline is H&E-only -- 5 tissue classes and nuclear morphometrics
# only, no protein-marker cell typing -- so that sub-feature is dropped
# entirely rather than approximated with a stain-only proxy.
_LESION_FEATURE_COLUMNS = (
    "instance_id",
    "region",
    "class_index",
    "class_name",
    "volume_um3",
    "volume_mm3",
    "centroid_z_um",
    "centroid_y_um",
    "centroid_x_um",
    "bbox_min_z_um",
    "bbox_min_y_um",
    "bbox_min_x_um",
    "bbox_max_z_um",
    "bbox_max_y_um",
    "bbox_max_x_um",
    "surface_area_um2",
    "equivalent_diameter_um",
    "extent",
    "euler_number",
    "feret_diameter_max_um",
    "axis_major_length_um",
    "axis_minor_length_um",
    "solidity",
    "size_class",
    # Seed voxel: a voxel index GUARANTEED to lie inside this lesion's own
    # connected component. `_lesion_mask` / `_assign_nuclei_to_lesions` re-find
    # the component by looking `lab` up here. They previously used the
    # CENTROID, which for a concave or sprawling 3D component frequently falls
    # outside the component itself -- measured on real data at 8 um isotropic,
    # 6,891 of 24,041 lesions' centroids landed on background and 10 landed on
    # a different component (28.7% misresolved). A seed voxel is preferred
    # over persisting the raw local component label because that numbering is
    # only valid for one exact (mask, connectivity) relabeling, whereas a seed
    # voxel stays correct under any relabeling.
    "seed_z_voxel",
    "seed_y_voxel",
    "seed_x_voxel",
)


# ---------------------------------------------------------------------------
# Degenerate-safe, memory-bounded shape descriptors for label_lesions.
#
# skimage's 3D regionprops feret_diameter_max/solidity/axis_major_length/
# axis_minor_length are all unsafe on real tissue geometry:
#   - feret_diameter_max: ValueError on a planar (single-z-slice) component
#     (marching_cubes needs a closed 3D surface to mesh); on a large
#     component it runs pdist over EVERY marching-cubes vertex of the hull
#     image -- quadratic in vertex count, observed as a 41.8-51.5 GB
#     allocation (SIGKILL) on real components.
#   - solidity: inf (area_convex == 0, divide by zero) on a planar
#     component.
#   - axis_minor_length: ValueError: math domain error (sqrt of a
#     negative inertia-eigenvalue sum) on some degenerate components.
# These module-private helpers reproduce skimage's exact numeric
# definitions on non-degenerate components (bit-identical for feret;
# 4-decimal-place agreement for solidity/axis lengths) while handling
# planar/collinear/single-voxel geometry without raising or blowing up
# memory. Validated against real data (see the quick-260826-fpn plan).
# ---------------------------------------------------------------------------

_VOXEL_CORNERS = np.array(
    [[a, b, c] for a in (-0.5, 0.5) for b in (-0.5, 0.5) for c in (-0.5, 0.5)],
    dtype=np.float64,
)

_FERET_DEGENERATE_PDIST_LIMIT = 5000


def _component_hull_image(mask: np.ndarray) -> np.ndarray | None:
    """Convex hull image of a bbox-cropped component mask, or `None` if
    degenerate.

    Computed ONCE per component and shared by BOTH the feret and solidity
    paths in `_component_shape_descriptors` -- on a large real bbox (e.g.
    27x397x480) this is the dominant per-component cost, so it must never
    be computed twice.

    Args:
        mask: `(z, y, x)` boolean array, bbox-cropped to one component.

    Returns:
        `(z, y, x)` boolean hull image, or `None` when `mask` is too
        degenerate for `convex_hull_image` to build a hull (planar,
        collinear, or single-voxel).
    """
    from scipy.spatial import QhullError
    from skimage.morphology import convex_hull_image

    try:
        return convex_hull_image(mask)
    except (ValueError, QhullError):
        return None


def _feret_degenerate_um(mask: np.ndarray, voxel_um: float) -> float:
    """Max caliper (feret) distance for a component with no 3D convex hull
    image (planar, collinear, or single-voxel).

    Expands each voxel centre to its 8 corners (`_VOXEL_CORNERS *
    voxel_um`) -- skimage measures shape corner-to-corner on the voxel
    IMAGE, not centre-to-centre, so this matches its convention. Above
    `_FERET_DEGENERATE_PDIST_LIMIT` expanded points, mean-centres, reduces
    to the point set's own rank via SVD, and hulls in that reduced
    subspace before re-expanding to the full corner set -- bounds
    `pdist`'s memory regardless of component size.

    Args:
        mask: `(z, y, x)` boolean array, bbox-cropped to one component.
        voxel_um: Isotropic voxel size in microns.

    Returns:
        Max caliper distance in microns; `0.0` for an all-`False` mask.
    """
    from scipy.spatial import ConvexHull, QhullError
    from scipy.spatial.distance import pdist

    pts = np.argwhere(mask).astype(np.float64) * voxel_um
    if len(pts) == 0:
        return 0.0

    expanded = (pts[:, None, :] + _VOXEL_CORNERS[None, :, :] * voxel_um).reshape(-1, 3)
    if len(expanded) > _FERET_DEGENERATE_PDIST_LIMIT:
        centred = expanded - expanded.mean(axis=0)
        _, sv, vt = np.linalg.svd(centred, full_matrices=False)
        tol = max(centred.shape) * np.finfo(float).eps * sv[0]
        rank = int((sv > tol).sum())
        reduced = centred @ vt[:rank].T
        try:
            expanded = expanded[ConvexHull(reduced).vertices]
        except QhullError:
            pass
    return float(np.sqrt(pdist(expanded, "sqeuclidean").max()))


def _feret_diameter_max_um(
    mask: np.ndarray, hull_image: np.ndarray | None, voxel_um: float
) -> float:
    """skimage's exact `feret_diameter_max` definition, with the `pdist`
    memory blowup removed.

    skimage runs `pdist` over EVERY marching-cubes vertex of the (padded)
    hull image -- quadratic in vertex count, observed as a 41.8-51.5 GB
    allocation (SIGKILL) on real components. The diameter of a point set
    EQUALS the diameter of its convex hull's vertices, so hulling the mesh
    vertices before `pdist` (1e5 verts -> a few hundred) is exact, not an
    approximation, and bounds the allocation regardless of component size.

    Performance (measured on the real crop this fix was written for): this
    costs ~7-8s each for the two largest real components (27x397x480 bbox)
    while most of the ~4,987 lesions in that crop are small and cheap, so
    `label_lesions` goes from ~2.8s (descriptors stubbed out) to roughly a
    minute overall. This is accepted and intentional -- do NOT "optimise"
    this away by sacrificing bit-exactness against skimage.

    Args:
        mask: `(z, y, x)` boolean array, bbox-cropped to one component.
        hull_image: `_component_hull_image(mask)`'s result, reused here
            rather than recomputed -- `None` routes straight to
            `_feret_degenerate_um`.
        voxel_um: Isotropic voxel size in microns.

    Returns:
        Max caliper distance in microns.
    """
    from scipy.spatial import ConvexHull, QhullError
    from scipy.spatial.distance import pdist
    from skimage.measure import marching_cubes

    if hull_image is None:
        return _feret_degenerate_um(mask, voxel_um)

    # skimage pads by 2 before meshing: the bbox crop has no background
    # margin, so the hull touches every face of the array and
    # marching_cubes raises "Surface level must be within volume data
    # range" without this pad. Required, not cosmetic.
    padded = np.pad(hull_image, 2, mode="constant", constant_values=False)
    try:
        verts, _, _, _ = marching_cubes(
            padded.astype(np.uint8), level=0.5, spacing=(voxel_um,) * 3
        )
    except (ValueError, QhullError):
        return _feret_degenerate_um(mask, voxel_um)
    if len(verts) < 2:
        return _feret_degenerate_um(mask, voxel_um)

    try:
        verts = verts[ConvexHull(verts).vertices]
    except QhullError:
        pass  # keep all verts -- still exact, just not memory-bounded
    return float(np.sqrt(pdist(verts, "sqeuclidean").max()))


def _degenerate_hull_volume_um3(mask: np.ndarray, voxel_um: float) -> float:
    """Convex hull volume (in um^3) of a degenerate component's voxel
    OUTER CORNERS -- the solidity-path counterpart to `_feret_degenerate_um`.

    A planar/collinear component still has a positive corner-hull volume
    (the Minkowski sum of the centre hull with the voxel cube is always
    3D), so solidity computed against this denominator stays finite --
    never `inf` the way `area_convex == 0` would make it (a flat plate
    genuinely IS convex; solidity 1.0 is correct).

    Args:
        mask: `(z, y, x)` boolean array, bbox-cropped to one component.
        voxel_um: Isotropic voxel size in microns.

    Returns:
        Hull volume in cubic microns; `float("nan")` if fewer than 4
        hull vertices result (caller must guard against this).
    """
    from scipy.spatial import ConvexHull, QhullError

    coords_um = np.argwhere(mask).astype(np.float64) * voxel_um
    pts = np.unique(coords_um, axis=0)
    if len(pts) == 0:
        return float("nan")
    if len(pts) == 1:
        core = pts
    else:
        centred = pts - pts.mean(axis=0)
        _, sv, vt = np.linalg.svd(centred, full_matrices=False)
        tol = max(centred.shape) * np.finfo(float).eps * (sv[0] if sv.size else 0.0)
        rank = int((sv > tol).sum())
        if rank <= 1:
            t = centred @ vt[0]
            core = pts[[int(t.argmin()), int(t.argmax())]]
        else:
            reduced = centred @ vt[:rank].T
            try:
                core = pts[ConvexHull(reduced).vertices]
            except QhullError:
                core = pts

    expanded = (core[:, None, :] + _VOXEL_CORNERS[None, :, :] * voxel_um).reshape(-1, 3)
    try:
        expanded = expanded[ConvexHull(expanded).vertices]
    except QhullError:
        pass
    if len(expanded) < 4:
        return float("nan")
    try:
        return float(ConvexHull(expanded).volume)
    except QhullError:
        return float("nan")


def _component_shape_descriptors(
    mask: np.ndarray, voxel_um: float, volume_um3: float
) -> dict[str, float]:
    """Degenerate-safe replacement for skimage's 3D
    `feret_diameter_max`/`solidity`/`axis_major_length`/`axis_minor_length`
    regionprops, computed directly from a component's bbox-cropped mask.

    Bit-identical to skimage on non-degenerate components (validated:
    feret absdiff == 0.0, solidity/axis lengths agree to 4 decimal
    places); never raises and never returns `inf`/`nan` on planar,
    collinear, or single-voxel components, which skimage's own
    regionprops cannot handle at all in 3D.

    `_component_hull_image(mask)` is called exactly ONCE here and its
    result reused for both the feret and solidity computations -- it is
    the dominant cost on a large bbox, so it must not be computed twice.

    Args:
        mask: `(z, y, x)` boolean array, bbox-cropped to one component
            (the same array already built for `_surface_area_um2`).
        voxel_um: Isotropic voxel size in microns.
        volume_um3: The component's already-known physical volume
            (`n_voxels * voxel_um**3`), reused as solidity's numerator
            rather than recomputed from `mask`.

    Returns:
        `dict` with exactly the keys `"feret_diameter_max_um"`,
        `"solidity"`, `"axis_major_length_um"`, `"axis_minor_length_um"`.
        `solidity` is never `inf`/`nan` (guarded, falls back to `1.0`).
    """
    from skimage.measure import inertia_tensor_eigvals

    hull_image = _component_hull_image(mask)
    feret_diameter_max_um = _feret_diameter_max_um(mask, hull_image, voxel_um)

    if hull_image is not None:
        # Exactly skimage's own definition -- unchanged on non-degenerate
        # components (validated to 4 dp: 0.9844 / 0.9964 / 0.9914).
        area_convex = float(hull_image.sum()) * voxel_um**3
    else:
        area_convex = _degenerate_hull_volume_um3(mask, voxel_um)

    if not np.isfinite(area_convex) or area_convex <= 0:
        solidity = 1.0
    else:
        solidity = volume_um3 / area_convex

    try:
        ev = np.sort(
            np.asarray(
                inertia_tensor_eigvals(mask, spacing=(voxel_um,) * 3),
                dtype=np.float64,
            )
        )[::-1]
    except TypeError:
        # Installed skimage rejects spacing= on this call -- inertia
        # tensor eigenvalues scale as length**2, so compute them in
        # voxel-index units and rescale by voxel_um**2 (exact for
        # isotropic voxels, which this module assumes throughout).
        ev = (
            np.sort(np.asarray(inertia_tensor_eigvals(mask), dtype=np.float64))[::-1]
            * voxel_um**2
        )

    # skimage: axis_major = sqrt(10*(ev[0]+ev[1]-ev[2])),
    # axis_minor = sqrt(10*(-ev[0]+ev[1]+ev[2])). Floating point can drive
    # either sum slightly negative on a degenerate component -- skimage
    # raises ValueError: math domain error there; clip to 0.0 instead (the
    # collapsed axis is genuinely zero-length).
    axis_major_length_um = float(
        np.sqrt(10.0 * np.clip(ev[0] + ev[1] - ev[2], 0.0, None))
    )
    axis_minor_length_um = float(
        np.sqrt(10.0 * np.clip(-ev[0] + ev[1] + ev[2], 0.0, None))
    )

    return {
        "feret_diameter_max_um": feret_diameter_max_um,
        "solidity": float(solidity),
        "axis_major_length_um": axis_major_length_um,
        "axis_minor_length_um": axis_minor_length_um,
    }


def label_lesions(
    sdata: "SpatialData",
    *,
    organ: str = "HGSC",
    voxel_um: float = 12.0,
    connectivity: int = 1,
    min_volume_um3: float = 1.0e4,
    size_threshold_um3: float = 1.0e6,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Per-lesion 3D connected-component feature table, across every
    non-space tissue class.

    Each non-space class is labeled independently, so a lesion never spans
    two classes. `connectivity=1` avoids merging corner-touching lesions
    into one. `min_volume_um3` filters segmentation noise before reporting;
    `size_threshold_um3` sets each surviving lesion's `size_class`
    (absolute threshold, not sample-relative).

    Args:
        sdata: `SpatialData` container with `labels["tissue_labels"]`.
        organ: Key into `config.TISSUE_CLASSES`. Defaults to `"HGSC"`.
        voxel_um: Isotropic voxel size, in microns. Must be `> 0`. Defaults
            to `12.0`.
        connectivity: `skimage.measure.label` connectivity, `(1, 2, 3)`.
            Defaults to `1`.
        min_volume_um3: Components smaller than this are dropped. Must be
            `>= 0`. Defaults to `1.0e4` (~5.8 voxels at 12 um).
        size_threshold_um3: `size_class` is `"large"` at/above this volume,
            else `"small"`. Must be `>= 0`. Defaults to `1.0e6`.
        csv_path: Optional path to also write the result as CSV.

    Returns:
        DataFrame with columns `_LESION_FEATURE_COLUMNS`, one row per
        surviving component. Also attached as `sdata.tables["lesions"]`
        (`TableModel`, `region="tissue_labels"`). Empty (but correctly
        shaped) when zero lesions survive.

    Raises:
        ValueError: `voxel_um <= 0`; `min_volume_um3 < 0`;
            `size_threshold_um3 < 0`; `connectivity` not in `(1, 2, 3)`.
        KeyError: `organ` unknown, or no `"tissue_labels"` element.
        ImportError: `anndata`/`spatialdata` are not installed.
    """
    _validate_positive("voxel_um", voxel_um)
    if min_volume_um3 < 0:
        raise ValueError(f"min_volume_um3 must be >= 0, got {min_volume_um3}")
    if size_threshold_um3 < 0:
        raise ValueError(f"size_threshold_um3 must be >= 0, got {size_threshold_um3}")
    if connectivity not in (1, 2, 3):
        raise ValueError(f"connectivity must be one of (1, 2, 3), got {connectivity}")

    class_map = _tissue_class_map(organ)
    non_space_classes = _non_space_class_indices(organ)

    from skimage.measure import label as sk_label
    from skimage.measure import regionprops_table

    labels_arr = _labels_array(sdata)

    rows: list[dict] = []
    for class_index in non_space_classes:
        class_name = class_map[class_index]
        mask = labels_arr == class_index
        lab, n_components = sk_label(mask, connectivity=connectivity, return_num=True)
        if n_components == 0:
            continue

        # Pre-filter by cheap voxel-count volume BEFORE calling
        # regionprops_table, rather than after: this is a compute
        # optimization, not a degeneracy dodge -- volume and planarity are
        # independent (the first real degenerate offender found was a
        # 57-voxel component spanning 8x11 voxels in a single z-plane,
        # substantial by volume, degenerate only in z; ~50% of surviving
        # components at 12 um isotropic occupy a single z-slice, measured
        # on a (40,480,480) crop: Epithelium 337/525, Stroma 581/1198,
        # rbc 739/1483, Intraluminal secretion 908/1781). Zeroing out
        # below-threshold labels here just means regionprops_table and the
        # expensive per-component shape-descriptor work never run on
        # segmentation noise that would be dropped anyway. Degeneracy
        # itself is handled unconditionally for every surviving component
        # by `_component_shape_descriptors`, not avoided by this filter.
        voxel_counts = np.bincount(lab.ravel(), minlength=n_components + 1)
        component_volumes_um3 = voxel_counts * voxel_um**3
        keep = np.zeros(n_components + 1, dtype=bool)
        keep[1:] = component_volumes_um3[1:] >= min_volume_um3
        if not keep.any():
            continue
        lab = np.where(keep[lab], lab, 0)

        table = regionprops_table(
            lab,
            properties=(
                "label",
                "area",
                "bbox",
                "centroid",
                "equivalent_diameter_area",
                "extent",
                "euler_number",
            ),
            spacing=(voxel_um,) * 3,
        )

        for i in range(len(table["label"])):
            volume_um3 = float(table["area"][i])
            if volume_um3 < min_volume_um3:
                continue

            local_label = int(table["label"][i])
            # regionprops_table's bbox is ALWAYS in raw voxel-index units
            # (`spacing=` scales area/centroid/lengths but NOT bbox), so no
            # um<->voxel back-conversion is needed here to slice `lab`.
            bbox_min = (
                int(table["bbox-0"][i]),
                int(table["bbox-1"][i]),
                int(table["bbox-2"][i]),
            )
            bbox_max = (
                int(table["bbox-3"][i]),
                int(table["bbox-4"][i]),
                int(table["bbox-5"][i]),
            )
            component_mask = (
                lab[
                    bbox_min[0] : bbox_max[0],
                    bbox_min[1] : bbox_max[1],
                    bbox_min[2] : bbox_max[2],
                ]
                == local_label
            )
            surface_area_um2 = _surface_area_um2(component_mask, voxel_um)
            descriptors = _component_shape_descriptors(
                component_mask, voxel_um, volume_um3
            )

            # First set voxel of the bbox-cropped mask, offset back to global
            # voxel coordinates. `component_mask` is non-empty by construction
            # (it came from a surviving regionprops row), so argwhere always
            # yields at least one index.
            seed_local = np.argwhere(component_mask)[0]
            seed_voxel = (
                int(bbox_min[0] + seed_local[0]),
                int(bbox_min[1] + seed_local[1]),
                int(bbox_min[2] + seed_local[2]),
            )

            rows.append(
                {
                    "region": "tissue_labels",
                    "class_index": class_index,
                    "class_name": class_name,
                    "volume_um3": volume_um3,
                    "volume_mm3": volume_um3 / 1e9,
                    "centroid_z_um": float(table["centroid-0"][i]),
                    "centroid_y_um": float(table["centroid-1"][i]),
                    "centroid_x_um": float(table["centroid-2"][i]),
                    "bbox_min_z_um": bbox_min[0] * voxel_um,
                    "bbox_min_y_um": bbox_min[1] * voxel_um,
                    "bbox_min_x_um": bbox_min[2] * voxel_um,
                    "bbox_max_z_um": bbox_max[0] * voxel_um,
                    "bbox_max_y_um": bbox_max[1] * voxel_um,
                    "bbox_max_x_um": bbox_max[2] * voxel_um,
                    "surface_area_um2": surface_area_um2,
                    "equivalent_diameter_um": float(
                        table["equivalent_diameter_area"][i]
                    ),
                    "extent": float(table["extent"][i]),
                    "euler_number": int(table["euler_number"][i]),
                    "feret_diameter_max_um": descriptors["feret_diameter_max_um"],
                    "axis_major_length_um": descriptors["axis_major_length_um"],
                    "axis_minor_length_um": descriptors["axis_minor_length_um"],
                    "solidity": descriptors["solidity"],
                    "size_class": "large" if volume_um3 >= size_threshold_um3 else "small",
                    "seed_z_voxel": seed_voxel[0],
                    "seed_y_voxel": seed_voxel[1],
                    "seed_x_voxel": seed_voxel[2],
                }
            )

    for instance_id, row in enumerate(rows, start=1):
        row["instance_id"] = instance_id

    df = pd.DataFrame(rows, columns=list(_LESION_FEATURE_COLUMNS))

    # Skip the sdata.tables["lesions"] attach step when zero lesions
    # survive: spatialdata 0.3.0's TableModel.validate() unconditionally
    # evaluates `obs[instance_key].iloc[0]` to dtype-check the instance
    # key, which raises IndexError on a 0-row AnnData (a spatialdata
    # library limitation, not a defect in this module's own construction).
    # The returned DataFrame's empty-but-correctly-schemad shape
    # (`list(_LESION_FEATURE_COLUMNS)`) is still produced either way; only
    # the SpatialData attach is skipped.
    if len(df) > 0:
        try:
            import anndata as ad
            from spatialdata.models import TableModel
        except ImportError as exc:
            raise ImportError(
                "anndata and spatialdata are required. "
                "Install: pip install anndata spatialdata"
            ) from exc

        obs = df.copy()
        obs.index = obs.index.astype(str)
        adata = ad.AnnData(obs=obs)
        adata.obsm["spatial_3d"] = df[
            ["centroid_z_um", "centroid_y_um", "centroid_x_um"]
        ].to_numpy(dtype=np.float64)
        sdata.tables["lesions"] = TableModel.parse(
            adata,
            region="tissue_labels",
            region_key="region",
            instance_key="instance_id",
        )

    _write_csv(df, csv_path)
    return df


# ---------------------------------------------------------------------------
# Per-lesion nuclear count/density/local tissue composition.
# ---------------------------------------------------------------------------


def _voxel_indices(
    coords_um: np.ndarray, voxel_um: float, shape: tuple[int, int, int]
) -> np.ndarray:
    """Convert `[z, y, x]` micron coordinates to integer voxel indices
    (floor division by `voxel_um`), clipped to `shape`. A pure index
    lookup, never a resampling/interpolation operation.

    Args:
        coords_um: `(N, 3)` float `[z, y, x]` micron coordinates.
        voxel_um: Isotropic voxel size, in microns.
        shape: The `(z, y, x)` shape of the label volume being indexed.

    Returns:
        `(N, 3)` int64 voxel indices, one row per input coordinate.
    """
    idx = np.floor(np.asarray(coords_um, dtype=np.float64) / voxel_um).astype(np.int64)
    upper = np.asarray(shape, dtype=np.int64) - 1
    return np.clip(idx, 0, upper)


_SEED_COLUMNS = ("seed_z_voxel", "seed_y_voxel", "seed_x_voxel")


def _resolve_component_label(
    lab: np.ndarray, row, voxel_um: float, *, lesion_id: int | None = None
) -> int:
    """Local connected-component label of the lesion described by `row`.

    Looks `lab` up at the row's persisted SEED VOXEL -- a voxel `label_lesions`
    recorded as being inside that component. Falls back to the centroid voxel
    only for a lesion table written before the seed columns existed.

    The centroid is NOT a safe locator: a concave or sprawling 3D component's
    centroid frequently sits outside the component. Measured on real 8 um
    isotropic data, 6,891 of 24,041 lesions' centroids landed on background
    and 10 landed on a different component -- 28.7% misresolved.

    Args:
        lab: `(z, y, x)` connected-component labeling of this lesion's class.
        row: A `sdata.tables["lesions"].obs` row.
        voxel_um: Isotropic voxel size, in microns.
        lesion_id: Reported in the error message; defaults to the row's own
            `instance_id` when present.

    Returns:
        The non-zero local component label.

    Raises:
        ValueError: The lookup yields 0. Label 0 is BACKGROUND, so returning it
            would make `lab == 0` select the whole background of the volume as
            the lesion -- silently, and for ~28.7% of real lesions under the
            old centroid lookup. Never treat 0 as a valid component.
    """
    if lesion_id is None and "instance_id" in row:
        lesion_id = int(row["instance_id"])

    if all(c in row for c in _SEED_COLUMNS) and not any(
        pd.isna(row[c]) for c in _SEED_COLUMNS
    ):
        idx = tuple(int(row[c]) for c in _SEED_COLUMNS)
        source = "seed voxel"
    else:
        centroid_um = np.array(
            [[row["centroid_z_um"], row["centroid_y_um"], row["centroid_x_um"]]]
        )
        idx = tuple(int(v) for v in _voxel_indices(centroid_um, voxel_um, lab.shape)[0])
        source = "centroid voxel (lesion table predates the seed columns)"

    local_label = int(lab[idx])
    if local_label == 0:
        raise ValueError(
            f"Lesion {lesion_id}: {source} {idx} resolves to component label 0 "
            "(background), so its component cannot be recovered. Label 0 is "
            "never a valid lesion. Re-run label_lesions() on this volume to "
            "regenerate the table with seed voxels, and make sure voxel_um and "
            "connectivity match the values it was called with."
        )
    return local_label


def _assign_nuclei_to_lesions(
    sdata: "SpatialData", *, organ: str, voxel_um: float, connectivity: int
) -> np.ndarray:
    """Map every nucleus in `sdata.tables["nuclei"]` to the lesion
    `instance_id` whose 3D connected component contains that nucleus's
    centroid voxel.

    Recomputes each class's connected-component labeling (cached per
    class), stamps every voxel of each lesion's component with its
    `instance_id` into one combined volume, then looks up each nucleus's
    centroid in that volume.

    Each lesion's component is recovered via `_resolve_component_label`
    (persisted seed voxel), NOT via its centroid. The old centroid lookup
    silently skipped every lesion whose centroid fell outside its own
    component -- 6,898 of 24,041 lesions got zero nuclei on real 8 um data,
    and 3 more were credited with a different lesion's nuclei.

    Args:
        sdata: A `SpatialData` container with `tables["lesions"]` (from a
            prior `label_lesions` call) and `tables["nuclei"]`.
        organ: Key into `config.TISSUE_CLASSES`.
        voxel_um: Isotropic voxel size, in microns. Must match the value
            `label_lesions` was called with.
        connectivity: `skimage.measure.label` connectivity. Must match the
            value `label_lesions` was called with.

    Returns:
        `(N,)` int64 array, one entry per nucleus (same row order as
        `sdata.tables["nuclei"].obsm["spatial_3d"]`); `0` means the
        nucleus's centroid voxel is not part of any surviving lesion.
    """
    from skimage.measure import label as sk_label

    labels_arr = _labels_array(sdata)
    lesions_obs = sdata.tables["lesions"].obs

    instance_volume = np.zeros(labels_arr.shape, dtype=np.int64)
    class_label_cache: dict[int, np.ndarray] = {}

    for _, row in lesions_obs.iterrows():
        class_index = int(row["class_index"])
        if class_index not in class_label_cache:
            mask = labels_arr == class_index
            lab, _ = sk_label(mask, connectivity=connectivity, return_num=True)
            class_label_cache[class_index] = lab
        lab = class_label_cache[class_index]

        local_label = _resolve_component_label(lab, row, voxel_um)
        instance_volume[lab == local_label] = int(row["instance_id"])

    nuclei_zyx_um = sdata.tables["nuclei"].obsm["spatial_3d"]
    nuclei_idx = _voxel_indices(nuclei_zyx_um, voxel_um, labels_arr.shape)
    return instance_volume[nuclei_idx[:, 0], nuclei_idx[:, 1], nuclei_idx[:, 2]]


def _local_tissue_composition(
    labels: np.ndarray,
    centroid_voxel_zyx: np.ndarray,
    radius_voxels: float,
    class_indices: tuple[int, ...],
) -> dict[int, float]:
    """Fraction of voxels within a sphere of `radius_voxels` around
    `centroid_voxel_zyx`, per class in `class_indices` (sums to 1.0).

    Crops a cube around the centroid first -- never scans the full volume.

    Args:
        labels: `(z, y, x)` tissue-class-index volume.
        centroid_voxel_zyx: `(3,)` float `[z, y, x]` centroid in voxel
            (not micron) units; rounded to the nearest voxel to define the
            sphere's center.
        radius_voxels: Sphere radius in voxel units.
        class_indices: Every class index to report a fraction for.

    Returns:
        `dict[int, float]` mapping each class index to its in-sphere
        voxel fraction. Every value is `0.0` in the (only possible when
        `radius_voxels <= 0`) edge case where the sphere contains zero
        voxels.
    """
    center = np.round(np.asarray(centroid_voxel_zyx, dtype=np.float64)).astype(np.int64)
    half_width = int(np.ceil(radius_voxels))
    shape = np.asarray(labels.shape)

    mins = np.clip(center - half_width, 0, None)
    maxs = np.clip(center + half_width + 1, None, shape)

    cropped = labels[mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]]

    zz, yy, xx = np.meshgrid(
        np.arange(mins[0], maxs[0]) - center[0],
        np.arange(mins[1], maxs[1]) - center[1],
        np.arange(mins[2], maxs[2]) - center[2],
        indexing="ij",
    )
    dist2 = (
        zz.astype(np.float64) ** 2
        + yy.astype(np.float64) ** 2
        + xx.astype(np.float64) ** 2
    )
    sphere_mask = dist2 <= radius_voxels**2

    total = int(sphere_mask.sum())
    if total == 0:
        return {c: 0.0 for c in class_indices}

    return {
        c: float(np.count_nonzero((cropped == c) & sphere_mask)) / total
        for c in class_indices
    }


def compute_lesion_cell_density(
    sdata: "SpatialData",
    *,
    organ: str = "HGSC",
    voxel_um: float = 12.0,
    connectivity: int = 1,
    composition_radius_um: float = 150.0,
    section_thickness_um: float = 4.0,
    sampling_interval: int = 1,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Per-lesion nuclear count, stereologically-corrected count, cell
    density, and local tissue composition, joined onto
    `sdata.tables["lesions"]` (from a prior `label_lesions` call).

    Nuclei are assigned to lesions via `_assign_nuclei_to_lesions`; the
    2D-to-3D count correction reuses `stereological_correct` directly.

    Args:
        sdata: `SpatialData` container with `tables["lesions"]` and
            `tables["nuclei"]`.
        organ, voxel_um, connectivity: Must match the `label_lesions` call
            that produced `tables["lesions"]`. Defaults `"HGSC"`, `12.0`, `1`.
        composition_radius_um: Local-sphere radius for `composition_frac_*`
            columns, in microns. Must be `> 0`. Defaults to `150.0`.
        section_thickness_um: Passed through to `stereological_correct`.
            Must be `> 0`. Defaults to `4.0`.
        sampling_interval: Passed through to `stereological_correct`. Must
            be `>= 1`. Defaults to `1`.
        csv_path: Optional path to also write the result as CSV.

    Returns:
        `sdata.tables["lesions"].obs`, mutated in place with new columns:
        `n_nuclei`, `n_nuclei_corrected`, `density_cells_per_mm3`,
        `density_corrected_cells_per_mm3`, and one `composition_frac_{c}`
        per class. A lesion with zero nuclei gets all-zero values.

    Raises:
        ValueError: `voxel_um <= 0`; `composition_radius_um <= 0`;
            `section_thickness_um <= 0`; `sampling_interval < 1`.
        KeyError: `"lesions"` or `"nuclei"` not in `sdata.tables`.
    """
    _validate_positive("voxel_um", voxel_um)
    _validate_positive("composition_radius_um", composition_radius_um)
    _validate_positive("section_thickness_um", section_thickness_um)
    if sampling_interval < 1:
        raise ValueError(
            "sampling_interval must be >= 1 (1 = every section imaged), got "
            f"{sampling_interval}"
        )

    if "lesions" not in sdata.tables:
        raise KeyError(
            "'lesions' not found in sdata.tables. Run label_lesions(sdata) "
            f"first. Available tables: {sorted(sdata.tables)}"
        )
    if "nuclei" not in sdata.tables:
        raise KeyError(
            f"'nuclei' not found in sdata.tables. Available tables: "
            f"{sorted(sdata.tables)}"
        )

    class_map = _tissue_class_map(organ)
    labels_arr = _labels_array(sdata)

    lesion_ids = _assign_nuclei_to_lesions(
        sdata, organ=organ, voxel_um=voxel_um, connectivity=connectivity
    )
    nuclei_obs = sdata.tables["nuclei"].obs
    lesions_obs = sdata.tables["lesions"].obs

    radius_voxels = composition_radius_um / voxel_um

    n_nuclei_col: list[int] = []
    n_nuclei_corrected_col: list[float] = []
    density_col: list[float] = []
    density_corrected_col: list[float] = []
    composition_cols: dict[int, list[float]] = {c: [] for c in class_map}

    for _, row in lesions_obs.iterrows():
        instance_id = int(row["instance_id"])
        subset = nuclei_obs.iloc[np.where(lesion_ids == instance_id)[0]]
        n_nuclei = len(subset)
        volume_mm3 = float(row["volume_mm3"])

        if n_nuclei == 0:
            n_nuclei_corrected = 0.0
            density = 0.0
            density_corrected = 0.0
        else:
            n_nuclei_corrected = stereological_correct(
                subset,
                section_thickness_um=section_thickness_um,
                sampling_interval=sampling_interval,
            )
            density = n_nuclei / volume_mm3
            density_corrected = n_nuclei_corrected / volume_mm3

        n_nuclei_col.append(n_nuclei)
        n_nuclei_corrected_col.append(n_nuclei_corrected)
        density_col.append(density)
        density_corrected_col.append(density_corrected)

        centroid_voxel = (
            np.array([row["centroid_z_um"], row["centroid_y_um"], row["centroid_x_um"]])
            / voxel_um
        )
        fractions = _local_tissue_composition(
            labels_arr, centroid_voxel, radius_voxels, tuple(class_map)
        )
        for c in class_map:
            composition_cols[c].append(fractions[c])

    lesions_obs["n_nuclei"] = n_nuclei_col
    lesions_obs["n_nuclei_corrected"] = n_nuclei_corrected_col
    lesions_obs["density_cells_per_mm3"] = density_col
    lesions_obs["density_corrected_cells_per_mm3"] = density_corrected_col
    for c in class_map:
        lesions_obs[f"composition_frac_{c}"] = composition_cols[c]

    _write_csv(lesions_obs, csv_path)
    return lesions_obs


# ---------------------------------------------------------------------------
# Global density/KDE/gradients, local-sphere queries, and per-nucleus
# local-neighborhood feature vectors.
# ---------------------------------------------------------------------------

# Kept configurable via a kwarg rather than hardcoded inside
# compute_neighborhood_features.
_NEIGHBORHOOD_RADII_UM = (50.0, 100.0, 200.0)


def _nucleus_class_indices(sdata: "SpatialData", voxel_um: float) -> np.ndarray:
    """Assign every nucleus in `sdata.tables["nuclei"]` the tissue-class
    index of the voxel its centroid falls in (via `_voxel_indices`).

    Args:
        sdata: A `SpatialData` container with `tables["nuclei"]` and
            `labels["tissue_labels"]`.
        voxel_um: Isotropic voxel size, in microns.

    Returns:
        `(N,)` int64 array, one tissue-class index per nucleus (same row
        order as `sdata.tables["nuclei"].obsm["spatial_3d"]`).

    Raises:
        KeyError: `"nuclei"` is not in `sdata.tables`, or `sdata.labels` has
            no `"tissue_labels"` element.
    """
    if "nuclei" not in sdata.tables:
        raise KeyError(
            f"'nuclei' not found in sdata.tables. Available tables: "
            f"{sorted(sdata.tables)}"
        )
    labels_arr = _labels_array(sdata)
    nuclei_zyx_um = sdata.tables["nuclei"].obsm["spatial_3d"]
    nuclei_idx = _voxel_indices(nuclei_zyx_um, voxel_um, labels_arr.shape)
    return labels_arr[nuclei_idx[:, 0], nuclei_idx[:, 1], nuclei_idx[:, 2]]


def density_grid(
    sdata: "SpatialData",
    *,
    voxel_um: float = 12.0,
    bandwidth_um: float = 50.0,
    return_gradient: bool = False,
):
    """Grid-binned 3D kernel density estimate of nuclear centroids.

    Bins centroids into a voxel histogram aligned to
    `labels["tissue_labels"]`'s grid, Gaussian-smooths when
    `bandwidth_um > 0`, and divides by `voxel_um**3` for cells/um3 units.
    Grid-binned (not an exact point-density estimator) since those don't
    scale past ~1e4 nuclei.

    Args:
        sdata: `SpatialData` container with `tables["nuclei"]` and
            `labels["tissue_labels"]`.
        voxel_um: Isotropic voxel size, in microns. Must be `> 0`. Defaults
            to `12.0`.
        bandwidth_um: Gaussian smoothing bandwidth, in microns. `0.0` skips
            smoothing. Must be `>= 0`. Defaults to `50.0`.
        return_gradient: Also return the gradient-magnitude array. Defaults
            to `False`.

    Returns:
        `(z, y, x)` float64 density array (cells/um3). If
        `return_gradient`, a 2-tuple `(density, gradient_magnitude)`.

    Raises:
        ValueError: `voxel_um <= 0`, or `bandwidth_um < 0`.
        KeyError: `"nuclei"` not in `sdata.tables`, or no
            `"tissue_labels"` element.
    """
    _validate_positive("voxel_um", voxel_um)
    if bandwidth_um < 0:
        raise ValueError(f"bandwidth_um must be >= 0, got {bandwidth_um}")
    if "nuclei" not in sdata.tables:
        raise KeyError(
            f"'nuclei' not found in sdata.tables. Available tables: "
            f"{sorted(sdata.tables)}"
        )

    from scipy.ndimage import gaussian_filter

    labels_arr = _labels_array(sdata)
    shape_zyx = labels_arr.shape
    centroids = np.asarray(sdata.tables["nuclei"].obsm["spatial_3d"], dtype=np.float64)

    hist, _ = np.histogramdd(
        centroids,
        bins=shape_zyx,
        range=[(0.0, s * voxel_um) for s in shape_zyx],
    )

    if bandwidth_um > 0:
        sigma_voxels = bandwidth_um / voxel_um
        hist = gaussian_filter(hist, sigma=sigma_voxels)

    density = hist / voxel_um**3

    if not return_gradient:
        return density

    grad = np.gradient(density, voxel_um)
    gradient_magnitude = np.linalg.norm(np.stack(grad), axis=0)
    return density, gradient_magnitude


def local_sphere_density(
    sdata: "SpatialData",
    points_zyx_um,
    *,
    organ: str = "HGSC",
    voxel_um: float = 12.0,
    radius_um: float = 150.0,
) -> pd.DataFrame:
    """Exact point-radius local cell density and tissue composition around
    one or more query points, via a single batched `cKDTree` query
    (never a grid-binned approximation).

    Args:
        sdata: `SpatialData` container with `tables["nuclei"]` and
            `labels["tissue_labels"]`.
        points_zyx_um: A `(3,)` `[z, y, x]` micron point, or `(M, 3)` points.
        organ: Key into `config.TISSUE_CLASSES`. Defaults to `"HGSC"`.
        voxel_um: Isotropic voxel size, in microns. Must be `> 0`. Defaults
            to `12.0`.
        radius_um: Sphere radius, in microns. Must be `> 0`. Defaults to
            `150.0`.

    Returns:
        `M`-row DataFrame with `n_nuclei`, `sphere_volume_mm3`,
        `density_cells_per_mm3`, and one `composition_frac_{c}` per class.

    Raises:
        ValueError: `voxel_um <= 0`, or `radius_um <= 0`.
        KeyError: `organ` unknown, `"nuclei"` not in `sdata.tables`, or no
            `"tissue_labels"` element.
    """
    _validate_positive("voxel_um", voxel_um)
    _validate_positive("radius_um", radius_um)
    class_map = _tissue_class_map(organ)
    if "nuclei" not in sdata.tables:
        raise KeyError(
            f"'nuclei' not found in sdata.tables. Available tables: "
            f"{sorted(sdata.tables)}"
        )

    from scipy.spatial import cKDTree

    labels_arr = _labels_array(sdata)
    centroids = np.asarray(sdata.tables["nuclei"].obsm["spatial_3d"], dtype=np.float64)

    points = np.atleast_2d(np.asarray(points_zyx_um, dtype=np.float64))

    tree = cKDTree(centroids)
    neighbor_lists = tree.query_ball_point(points, r=radius_um)

    sphere_volume_mm3 = (4.0 / 3.0) * np.pi * (radius_um / 1000.0) ** 3
    radius_voxels = radius_um / voxel_um

    rows = []
    for point, neighbor_idx in zip(points, neighbor_lists):
        n_nuclei = len(neighbor_idx)
        density_cells_per_mm3 = (
            n_nuclei / sphere_volume_mm3 if sphere_volume_mm3 > 0 else 0.0
        )
        centroid_voxel = point / voxel_um
        fractions = _local_tissue_composition(
            labels_arr, centroid_voxel, radius_voxels, tuple(class_map)
        )
        row = {
            "n_nuclei": n_nuclei,
            "sphere_volume_mm3": sphere_volume_mm3,
            "density_cells_per_mm3": density_cells_per_mm3,
        }
        for c in class_map:
            row[f"composition_frac_{c}"] = fractions[c]
        rows.append(row)

    return pd.DataFrame(rows)


def compute_density(
    sdata: "SpatialData",
    *,
    organ: str = "HGSC",
    voxel_um: float = 12.0,
    bandwidth_um: float = 50.0,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Per-tissue-class global cell density table: nuclei per class
    (exact voxel lookup) over that class's physical volume, in cells/mm3.

    Args:
        sdata: `SpatialData` container with `tables["nuclei"]` and
            `labels["tissue_labels"]`.
        organ: Key into `config.TISSUE_CLASSES`. Defaults to `"HGSC"`.
        voxel_um: Isotropic voxel size, in microns. Must be `> 0`.
            Defaults to `12.0`.
        bandwidth_um: Accepted for signature symmetry with `density_grid`
            only; unused here. Must be `>= 0`. Defaults to `50.0`.
        csv_path: Optional path to also write the result as CSV.

    Returns:
        DataFrame with `class_index`, `class_name`, `n_nuclei`,
        `class_volume_mm3`, `density_cells_per_mm3` (`0.0`, not `NaN`, when
        volume is `0`), one row per class. Also attached in place at
        `sdata.tables["nuclei"].uns["density"]`.

    Raises:
        ValueError: `voxel_um <= 0`, or `bandwidth_um < 0`.
        KeyError: `organ` unknown, `"nuclei"` not in `sdata.tables`, or no
            `"tissue_labels"` element.
    """
    _validate_positive("voxel_um", voxel_um)
    if bandwidth_um < 0:
        raise ValueError(f"bandwidth_um must be >= 0, got {bandwidth_um}")
    class_map = _tissue_class_map(organ)
    if "nuclei" not in sdata.tables:
        raise KeyError(
            f"'nuclei' not found in sdata.tables. Available tables: "
            f"{sorted(sdata.tables)}"
        )

    labels_arr = _labels_array(sdata)
    nucleus_classes = _nucleus_class_indices(sdata, voxel_um)

    rows = []
    for class_index, class_name in class_map.items():
        n_voxels = int(np.count_nonzero(labels_arr == class_index))
        class_volume_mm3 = (n_voxels * voxel_um**3) / 1e9
        n_nuclei = int(np.count_nonzero(nucleus_classes == class_index))
        density_cells_per_mm3 = (
            n_nuclei / class_volume_mm3 if class_volume_mm3 > 0 else 0.0
        )
        rows.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "n_nuclei": n_nuclei,
                "class_volume_mm3": class_volume_mm3,
                "density_cells_per_mm3": density_cells_per_mm3,
            }
        )

    df = pd.DataFrame(rows)
    sdata.tables["nuclei"].uns["density"] = df

    _write_csv(df, csv_path)
    return df


def compute_neighborhood_features(
    sdata: "SpatialData",
    *,
    organ: str = "HGSC",
    voxel_um: float = 12.0,
    radii_um: tuple[float, ...] = _NEIGHBORHOOD_RADII_UM,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Per-nucleus local-neighborhood feature vector -- extends
    `sdata.tables["nuclei"].obs` in place. Feeds downstream niche-discovery
    clustering (`sc.pp.neighbors`/`sc.tl.leiden`).

    For each radius `R` in `radii_um` (columns suffixed `_r{int(R)}`):
      - `n_neighbors_r{R}`: other nuclei within `R` microns (self excluded).
      - `density_cells_per_mm3_r{R}`: that count over the sphere volume.
      - `composition_frac_{c}_r{R}`: fraction of nuclei within `R` (self
        included) in class `c`, per class.
      - `shannon_entropy_r{R}`: entropy of those composition fractions.
    Radius-independent: `nn_dist_um_class_{c}`, distance to the nearest
    OTHER nucleus of class `c` (`np.nan` if no such neighbor exists).

    To persist these columns to an already-written Zarr store, call
    `_persist_element(sdata, "nuclei")` afterward.

    Args:
        sdata: `SpatialData` container with `tables["nuclei"]` and
            `labels["tissue_labels"]`.
        organ: Key into `config.TISSUE_CLASSES`. Defaults to `"HGSC"`.
        voxel_um: Isotropic voxel size, in microns. Must be `> 0`.
            Defaults to `12.0`.
        radii_um: Radii, in microns, for the per-radius columns. Every
            entry must be `> 0`. Defaults to `(50.0, 100.0, 200.0)`.
        csv_path: Optional path to also write just the new columns as CSV.

    Returns:
        DataFrame of only the newly-added columns (same row order as
        `sdata.tables["nuclei"].obs`), which is also mutated in place.

    Raises:
        ValueError: `voxel_um <= 0`, or any `radii_um` entry `<= 0`.
        KeyError: `organ` unknown, `"nuclei"` not in `sdata.tables`, or no
            `"tissue_labels"` element.
    """
    _validate_positive("voxel_um", voxel_um)
    for i, r_um in enumerate(radii_um):
        if r_um <= 0:
            raise ValueError(f"radii_um[{i}] must be > 0, got {r_um}")
    class_map = _tissue_class_map(organ)
    if "nuclei" not in sdata.tables:
        raise KeyError(
            f"'nuclei' not found in sdata.tables. Available tables: "
            f"{sorted(sdata.tables)}"
        )

    from scipy.spatial import cKDTree

    nuclei_obs = sdata.tables["nuclei"].obs
    centroids = np.asarray(sdata.tables["nuclei"].obsm["spatial_3d"], dtype=np.float64)
    n_nuclei = centroids.shape[0]
    nucleus_classes = _nucleus_class_indices(sdata, voxel_um)

    tree = cKDTree(centroids)

    # One cKDTree per tissue class, built once and reused across every
    # radius -- skip building a tree for a class with zero nuclei (its
    # nn_dist_um_class_{c} column is np.nan for every nucleus).
    class_trees: dict[int, tuple[cKDTree, np.ndarray]] = {}
    for class_index in class_map:
        member_idx = np.where(nucleus_classes == class_index)[0]
        if member_idx.size > 0:
            class_trees[class_index] = (cKDTree(centroids[member_idx]), member_idx)

    new_columns: dict[str, np.ndarray] = {}

    for r_um in radii_um:
        r_key = int(r_um)
        sphere_volume_mm3 = (4.0 / 3.0) * np.pi * (r_um / 1000.0) ** 3

        # COUNT-ONLY queries (return_length=True): scipy returns one integer
        # per query point and never materialises the neighbour index lists.
        # The previous implementation called query_ball_point WITHOUT
        # return_length and then looped over the lists in Python purely to
        # derive these same counts. Because the nuclei fill a fixed volume,
        # per-nucleus neighbour count grows linearly with N, so materialising
        # the lists costs O(N^2) memory. Measured on the real 8 um store:
        # 100k nuclei -> 1.06e7 list entries / 2.2 GB / 28.9 s; 400k ->
        # 1.68e8 entries / 6.1 GB / 102.7 s (15.9x entries for 4x N). At the
        # real 9,522,193 nuclei that extrapolates to ~9.6e10 entries and
        # ~2.7 TB -- the function simply could not run. Counting instead is
        # O(N) in memory and drops the 9.5M-iteration Python loop entirely.
        total_in_radius = np.asarray(
            tree.query_ball_point(centroids, r=r_um, return_length=True),
            dtype=np.int64,
        )
        # query_ball_point always includes the query point itself (distance
        # 0.0 <= r_um for any r_um > 0), so total_in_radius includes self and
        # n_neighbors excludes it -- unchanged from the previous behaviour.
        n_neighbors = total_in_radius - 1
        density = n_neighbors / sphere_volume_mm3

        # Per-class counts from the SAME per-class trees already built above
        # for nn_dist_um_class_{c}: 5 classes x 3 radii = 15 count-only
        # queries, versus one giant list allocation.
        entropy = np.zeros(n_nuclei, dtype=np.float64)
        frac_cols: dict[int, np.ndarray] = {}
        for class_index in class_map:
            if class_index in class_trees:
                class_tree, _ = class_trees[class_index]
                class_counts = np.asarray(
                    class_tree.query_ball_point(
                        centroids, r=r_um, return_length=True
                    ),
                    dtype=np.int64,
                )
            else:
                # No nuclei of this class anywhere -- fraction is 0.0 for
                # every nucleus, matching the old per-nucleus count of zero.
                class_counts = np.zeros(n_nuclei, dtype=np.int64)

            frac = class_counts / total_in_radius
            frac_cols[class_index] = frac
            # Vectorised -x*log(x), skipping frac == 0 exactly as the old
            # `if frac > 0.0` branch did (0*log 0 contributes nothing).
            nonzero = frac > 0.0
            entropy[nonzero] -= frac[nonzero] * np.log(frac[nonzero])

        new_columns[f"n_neighbors_r{r_key}"] = n_neighbors
        new_columns[f"density_cells_per_mm3_r{r_key}"] = density
        for class_index in class_map:
            new_columns[f"composition_frac_{class_index}_r{r_key}"] = frac_cols[
                class_index
            ]
        new_columns[f"shannon_entropy_r{r_key}"] = entropy

    for class_index in class_map:
        col = np.full(n_nuclei, np.nan, dtype=np.float64)
        if class_index in class_trees:
            class_tree, member_idx = class_trees[class_index]
            # k=2 uniformly (rather than branching per-nucleus) so a
            # nucleus that IS a member of this class skips its own
            # self-match (index 0, distance 0.0) via the second column;
            # scipy pads missing neighbours with inf when class_tree has
            # fewer than 2 points, which naturally becomes np.nan below.
            dists, _ = class_tree.query(centroids, k=2)
            is_member = np.zeros(n_nuclei, dtype=bool)
            is_member[member_idx] = True
            col = np.where(is_member, dists[:, 1], dists[:, 0])
            col = np.where(np.isinf(col), np.nan, col)
        new_columns[f"nn_dist_um_class_{class_index}"] = col

    new_df = pd.DataFrame(new_columns, index=nuclei_obs.index)

    for col_name, values in new_columns.items():
        nuclei_obs[col_name] = values

    _write_csv(new_df, csv_path)
    return new_df


# ---------------------------------------------------------------------------
# Structure-level radial (concentric-ring) profiling and surface-tangential
# geodesic decorrelation length.
# ---------------------------------------------------------------------------


def _crop_with_padding(
    volume: np.ndarray, bbox_voxels: tuple[int, ...], pad_voxels: int
) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    """Crop `volume` to `bbox_voxels` padded by `pad_voxels` on every side,
    clamped to `volume`'s own bounds. Used before any
    `distance_transform_edt`/`marching_cubes` call -- never on the full
    organ-scale volume.

    Args:
        volume: `(z, y, x)` ndarray to crop (label volume or bool mask).
        bbox_voxels: `(zmin, ymin, xmin, zmax, ymax, xmax)`, min inclusive,
            max exclusive (regionprops_table convention).
        pad_voxels: Padding to add per side before clamping to `volume.shape`.

    Returns:
        `(cropped, slices)` -- `cropped` is a view (not a copy); `slices`
        lets callers recover the crop's origin (`slices[i].start`).
    """
    zmin, ymin, xmin, zmax, ymax, xmax = bbox_voxels
    shape = volume.shape
    z0 = max(int(zmin) - pad_voxels, 0)
    y0 = max(int(ymin) - pad_voxels, 0)
    x0 = max(int(xmin) - pad_voxels, 0)
    z1 = min(int(zmax) + pad_voxels, shape[0])
    y1 = min(int(ymax) + pad_voxels, shape[1])
    x1 = min(int(xmax) + pad_voxels, shape[2])
    slices = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    return volume[slices], slices


def _lesion_bbox_voxels(row, voxel_um: float) -> tuple[int, int, int, int, int, int]:
    """Convert a `sdata.tables["lesions"].obs` row's `bbox_*_um` columns
    back to voxel indices, matching `label_lesions`'s own
    `regionprops_table(..., spacing=(voxel_um,)*3)` convention exactly
    (voxel index * voxel_um, never a "+0.5 voxel-center" offset)."""
    return (
        round(row["bbox_min_z_um"] / voxel_um),
        round(row["bbox_min_y_um"] / voxel_um),
        round(row["bbox_min_x_um"] / voxel_um),
        round(row["bbox_max_z_um"] / voxel_um),
        round(row["bbox_max_y_um"] / voxel_um),
        round(row["bbox_max_x_um"] / voxel_um),
    )


def _lesion_mask(
    sdata: "SpatialData",
    lesion_id: int,
    *,
    organ: str,
    voxel_um: float,
    connectivity: int,
):
    """Rebuild the full-volume boolean mask for one lesion's exact 3D
    connected component, via its recorded seed voxel -- never re-running
    `regionprops_table`'s shape-descriptor properties.

    Uses `_resolve_component_label`, which raises rather than ever letting a
    background lookup through. The previous centroid-based lookup returned
    `lab == 0` -- the ENTIRE BACKGROUND of the volume -- whenever a lesion's
    centroid fell outside its own component: measured at 411,748,935 voxels
    returned for a 12,952,448-voxel lesion (31.8x), with `radial_profile`
    then producing a complete, plausible-looking profile from it.

    Args:
        sdata: `SpatialData` container with `tables["lesions"]` (from a
            prior `label_lesions` call).
        lesion_id: The `instance_id` to look up.
        organ: Key into `config.TISSUE_CLASSES`; validated but otherwise
            unused (the lesion's own `class_index` selects the mask).
        voxel_um, connectivity: Must match the `label_lesions` call.

    Returns:
        `(structure_mask, lesion_row)` -- boolean `(z, y, x)` mask, and
        the matching `pandas.Series` from `tables["lesions"].obs`.

    Raises:
        KeyError: `organ` unknown; `"lesions"` not in `sdata.tables`; or
            `lesion_id` not found.
        ValueError: the lesion's component could not be recovered (its
            locator voxel resolves to background) -- see
            `_resolve_component_label`.
    """
    _tissue_class_map(organ)  # validates organ; class_index below is what
    # actually selects the mask.

    if "lesions" not in sdata.tables:
        raise KeyError(
            "'lesions' not found in sdata.tables. Run label_lesions(sdata) "
            f"first. Available tables: {sorted(sdata.tables)}"
        )
    lesions_obs = sdata.tables["lesions"].obs
    matches = lesions_obs.index[lesions_obs["instance_id"] == lesion_id]
    if len(matches) == 0:
        available = sorted(int(v) for v in lesions_obs["instance_id"])
        raise KeyError(
            f"lesion_id {lesion_id!r} not found in sdata.tables['lesions']. "
            f"Available instance_id values: {available}"
        )
    row = lesions_obs.loc[matches[0]]
    class_index = int(row["class_index"])

    from skimage.measure import label as sk_label

    labels_arr = _labels_array(sdata)
    mask = labels_arr == class_index
    lab, _ = sk_label(mask, connectivity=connectivity, return_num=True)

    local_label = _resolve_component_label(lab, row, voxel_um, lesion_id=lesion_id)
    structure_mask = lab == local_label
    return structure_mask, row


def radial_profile(
    sdata: "SpatialData",
    lesion_id: int,
    *,
    organ: str = "HGSC",
    voxel_um: float = 12.0,
    ring_width_um: float = 10.0,
    max_radius_um: float = 500.0,
    proximal_um: float = 50.0,
    distal_um: float = 200.0,
    connectivity: int = 1,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Structure-level radial (concentric-ring) microenvironment profile.

    Shell distances come from a true 3D `distance_transform_edt` (never a
    per-section 2D approximation), computed on a bbox-cropped,
    `max_radius_um`-padded subvolume -- never the full label volume (a
    full-organ EDT would be a multi-GB allocation). Shells are `(r_lo,
    r_hi]` intervals, so the structure's own interior (distance `0.0`) is
    never counted as its own zero-radius shell.

    Args:
        sdata: `SpatialData` container with `labels["tissue_labels"]` and
            `tables["lesions"]` (from a prior `label_lesions` call).
        lesion_id: The `instance_id` to profile.
        organ, voxel_um, connectivity: Must match the `label_lesions` call.
            Defaults `"HGSC"`, `12.0`, `1`.
        ring_width_um: Shell thickness, in microns. Must be `> 0`. Defaults
            to `10.0`.
        max_radius_um: Maximum profiling radius, in microns. Must be
            `>= ring_width_um`. Defaults to `500.0`.
        proximal_um: Shells with `r_hi_um <= proximal_um` aggregate into
            the `"proximal"` summary row. Must be `> 0`. Defaults to `50.0`.
        distal_um: Shells with `r_lo_um >= distal_um` aggregate into the
            `"distal"` summary row. Must be `>= proximal_um`. Defaults to
            `200.0`.
        csv_path: Optional path to also write the result as CSV.

    Returns:
        DataFrame: one row per shell (`band="shell"`, `shell_index`,
        `r_lo_um`/`r_hi_um`, `n_voxels`, `n_nuclei`, `density_cells_per_mm3`,
        one `composition_frac_{c}` per class), plus two summary rows
        (`band in {"proximal", "distal"}`) aggregating the qualifying
        shells. `np.nan` (not dropped) for an empty shell's fractions.

    Raises:
        ValueError: `voxel_um <= 0`; `ring_width_um <= 0`;
            `max_radius_um <= 0` or `< ring_width_um`; `proximal_um <= 0`;
            `distal_um <= 0` or `< proximal_um`.
        KeyError: `organ` is not a key of `config.TISSUE_CLASSES`;
            `"lesions"` is not in `sdata.tables`; or `lesion_id` is not one
            of `sdata.tables["lesions"].obs["instance_id"]`.
    """
    _validate_positive("voxel_um", voxel_um)
    _validate_positive("ring_width_um", ring_width_um)
    _validate_positive("max_radius_um", max_radius_um)
    _validate_positive("proximal_um", proximal_um)
    _validate_positive("distal_um", distal_um)
    if max_radius_um < ring_width_um:
        raise ValueError(
            f"max_radius_um ({max_radius_um}) must be >= ring_width_um "
            f"({ring_width_um})"
        )
    if distal_um < proximal_um:
        raise ValueError(
            f"distal_um ({distal_um}) must be >= proximal_um ({proximal_um})"
        )

    class_map = _tissue_class_map(organ)

    structure_mask, row = _lesion_mask(
        sdata, lesion_id, organ=organ, voxel_um=voxel_um, connectivity=connectivity
    )

    labels_arr = _labels_array(sdata)
    pad_voxels = int(np.ceil(max_radius_um / voxel_um))
    bbox_voxels = _lesion_bbox_voxels(row, voxel_um)
    cropped_labels, crop_slices = _crop_with_padding(
        labels_arr, bbox_voxels, pad_voxels
    )
    structure_mask_cropped = structure_mask[crop_slices]

    from scipy.ndimage import distance_transform_edt

    dist_um = distance_transform_edt(~structure_mask_cropped, sampling=(voxel_um,) * 3)

    edges = np.arange(0.0, max_radius_um + ring_width_um, ring_width_um)
    n_shells = len(edges) - 1

    flat_dist = dist_um.ravel()
    flat_labels = cropped_labels.ravel()
    # (r_lo, r_hi] shells: searchsorted(..., side="left") - 1 maps a
    # distance exactly ON an edge to the LOWER shell (right-inclusive),
    # and maps distance == 0.0 (the structure's own interior) to shell
    # index -1 -- automatically excluded below, never double-counted as
    # its own zero-radius shell.
    shell_idx_all = np.searchsorted(edges, flat_dist, side="left") - 1

    origin = np.array([s.start for s in crop_slices])
    if "nuclei" in sdata.tables:
        nuclei_zyx_um = np.asarray(
            sdata.tables["nuclei"].obsm["spatial_3d"], dtype=np.float64
        )
        nuclei_full_idx = _voxel_indices(nuclei_zyx_um, voxel_um, labels_arr.shape)
        local_idx = nuclei_full_idx - origin
        crop_shape = np.array(dist_um.shape)
        in_crop = np.all((local_idx >= 0) & (local_idx < crop_shape), axis=1)
        nucleus_shell = np.full(nuclei_zyx_um.shape[0], -1, dtype=np.int64)
        if in_crop.any():
            li = local_idx[in_crop]
            d = dist_um[li[:, 0], li[:, 1], li[:, 2]]
            nucleus_shell[in_crop] = np.searchsorted(edges, d, side="left") - 1
    else:
        nucleus_shell = np.empty(0, dtype=np.int64)

    class_indices = tuple(class_map)
    shell_rows: list[dict] = []
    shell_voxel_counts: list[int] = []
    shell_class_counts: list[dict[int, int]] = []
    shell_n_nuclei: list[int] = []

    for shell_index in range(n_shells):
        r_lo = float(edges[shell_index])
        r_hi = float(edges[shell_index + 1])
        mask_shell = shell_idx_all == shell_index
        n_voxels = int(mask_shell.sum())
        n_nuclei_shell = int(np.count_nonzero(nucleus_shell == shell_index))

        row_out = {
            "shell_index": shell_index,
            "r_lo_um": r_lo,
            "r_hi_um": r_hi,
            "band": "shell",
            "n_voxels": n_voxels,
            "n_nuclei": n_nuclei_shell,
        }
        class_counts: dict[int, int] = {}
        if n_voxels > 0:
            labels_in_shell = flat_labels[mask_shell]
            for c in class_indices:
                count_c = int(np.count_nonzero(labels_in_shell == c))
                class_counts[c] = count_c
                row_out[f"composition_frac_{c}"] = count_c / n_voxels
            shell_volume_mm3 = n_voxels * voxel_um**3 / 1e9
            row_out["density_cells_per_mm3"] = (
                n_nuclei_shell / shell_volume_mm3 if shell_volume_mm3 > 0 else 0.0
            )
        else:
            for c in class_indices:
                class_counts[c] = 0
                row_out[f"composition_frac_{c}"] = np.nan
            row_out["density_cells_per_mm3"] = np.nan

        shell_rows.append(row_out)
        shell_voxel_counts.append(n_voxels)
        shell_class_counts.append(class_counts)
        shell_n_nuclei.append(n_nuclei_shell)

    def _band_row(band_name: str, keep: list[bool]) -> dict:
        total_voxels = sum(v for v, k in zip(shell_voxel_counts, keep) if k)
        total_nuclei = sum(v for v, k in zip(shell_n_nuclei, keep) if k)
        row_out = {
            "shell_index": -1,
            "r_lo_um": np.nan,
            "r_hi_um": np.nan,
            "band": band_name,
            "n_voxels": total_voxels,
            "n_nuclei": total_nuclei,
        }
        if total_voxels > 0:
            for c in class_indices:
                count_c = sum(
                    counts[c] for counts, k in zip(shell_class_counts, keep) if k
                )
                row_out[f"composition_frac_{c}"] = count_c / total_voxels
            volume_mm3 = total_voxels * voxel_um**3 / 1e9
            row_out["density_cells_per_mm3"] = (
                total_nuclei / volume_mm3 if volume_mm3 > 0 else 0.0
            )
        else:
            for c in class_indices:
                row_out[f"composition_frac_{c}"] = np.nan
            row_out["density_cells_per_mm3"] = np.nan
        return row_out

    proximal_keep = [r["r_hi_um"] <= proximal_um for r in shell_rows]
    distal_keep = [r["r_lo_um"] >= distal_um for r in shell_rows]
    proximal_row = _band_row("proximal", proximal_keep)
    proximal_row["r_lo_um"] = 0.0
    proximal_row["r_hi_um"] = proximal_um
    distal_row = _band_row("distal", distal_keep)
    distal_row["r_lo_um"] = distal_um
    distal_row["r_hi_um"] = max_radius_um

    df = pd.DataFrame(shell_rows + [proximal_row, distal_row])
    _write_csv(df, csv_path)
    return df


def _surface_geodesic_distances(
    verts: np.ndarray, faces: np.ndarray, source_vertex_idx: int
) -> np.ndarray:
    """Discrete surface geodesic distances via mesh-graph Dijkstra.

    Args:
        verts: `(n_verts, 3)` mesh vertex coordinates, in microns.
        faces: `(n_faces, 3)` triangle vertex-index array (from
            `skimage.measure.marching_cubes`).
        source_vertex_idx: Index into `verts` to compute distances from.

    Returns:
        `(n_verts,)` float array of geodesic distances, in microns, along
        mesh edges only.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    n = verts.shape[0]
    graph = coo_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(n, n))
    graph = graph.maximum(graph.T)
    return dijkstra(graph, directed=False, indices=[source_vertex_idx])[0]


def surface_decorrelation(
    sdata: "SpatialData",
    lesion_id: int,
    *,
    organ: str = "HGSC",
    voxel_um: float = 12.0,
    metric: str | np.ndarray = "cell_density",
    metric_radius_um: float = 100.0,
    thresholds: tuple[float, ...] = (0.25, 0.50, 1.00),
    source_vertex: int | None = None,
    connectivity: int = 1,
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Surface-tangential decorrelation length along a structure's 3D
    surface: the geodesic distance from a source vertex at which a metric's
    relative change first crosses each threshold.

    Deliberate divergence: geodesic distance is a discrete mesh-graph
    shortest path, not a dedicated heat-method geodesic-mesh library.

    Args:
        sdata: `SpatialData` container with `labels["tissue_labels"]`,
            `tables["lesions"]`, and (if `metric == "cell_density"`)
            `tables["nuclei"]`.
        lesion_id: The `instance_id` to profile.
        organ, voxel_um, connectivity: Must match the `label_lesions` call.
            Defaults `"HGSC"`, `12.0`, `1`.
        metric: `"cell_density"` (samples `local_sphere_density` at each
            surface vertex) or an `(n_surface_vertices,)` ndarray in
            mesh-vertex order.
        metric_radius_um: Radius for `local_sphere_density` when
            `metric == "cell_density"`. Must be `> 0`. Defaults to `100.0`.
        thresholds: Relative-change thresholds to report. Every entry must
            be `> 0`. Defaults to `(0.25, 0.50, 1.00)`.
        source_vertex: Vertex to measure geodesic distance from. `None`
            (default) picks the max-metric vertex.
        csv_path: Optional path to also write the result as CSV.

    Returns:
        One row per threshold: `threshold`, `decorrelation_um` (`np.nan` if
        never crossed), `lesion_id`, `n_surface_vertices`,
        `source_vertex_index`, `metric_at_source`, plus broadcast
        `decorrelation_um_{pct}` accessor columns. Falls back to absolute
        (not relative) change when `metric_at_source == 0`.

    Raises:
        ValueError: `voxel_um <= 0`; `metric_radius_um <= 0`; any entry of
            `thresholds` is `<= 0` (message names the offending value and
            its index); `metric` is a string other than `"cell_density"`;
            `metric` is an ndarray whose shape does not match the number
            of surface vertices.
        KeyError: `organ` is not a key of `config.TISSUE_CLASSES`;
            `"lesions"` is not in `sdata.tables`; `lesion_id` is not one of
            `sdata.tables["lesions"].obs["instance_id"]`; or
            `metric == "cell_density"` and `"nuclei"` is not in
            `sdata.tables`.
    """
    _validate_positive("voxel_um", voxel_um)
    _validate_positive("metric_radius_um", metric_radius_um)
    for i, t in enumerate(thresholds):
        if t <= 0:
            raise ValueError(f"thresholds[{i}] must be > 0, got {t}")

    structure_mask, row = _lesion_mask(
        sdata, lesion_id, organ=organ, voxel_um=voxel_um, connectivity=connectivity
    )

    bbox_voxels = _lesion_bbox_voxels(row, voxel_um)
    cropped_mask, crop_slices = _crop_with_padding(
        structure_mask, bbox_voxels, pad_voxels=1
    )

    from skimage.measure import marching_cubes

    verts_local, faces, _, _ = marching_cubes(
        cropped_mask.astype(np.uint8), level=0.5, spacing=(voxel_um,) * 3
    )
    origin_um = np.array([s.start for s in crop_slices]) * voxel_um
    verts = verts_local + origin_um
    n_verts = verts.shape[0]

    if isinstance(metric, np.ndarray):
        if metric.shape != (n_verts,):
            raise ValueError(
                f"metric array shape {metric.shape} does not match the "
                f"number of surface vertices ({n_verts},)"
            )
        metric_values = np.asarray(metric, dtype=np.float64)
    elif metric == "cell_density":
        if "nuclei" not in sdata.tables:
            raise KeyError(
                "'nuclei' not found in sdata.tables (required for "
                "metric='cell_density'). Available tables: "
                f"{sorted(sdata.tables)}"
            )
        density_df = local_sphere_density(
            sdata, verts, organ=organ, voxel_um=voxel_um, radius_um=metric_radius_um
        )
        metric_values = density_df["density_cells_per_mm3"].to_numpy(dtype=np.float64)
    else:
        raise ValueError(
            f"Unsupported metric: {metric!r}. Supported options: "
            "'cell_density', or an (n_surface_vertices,) numpy array."
        )

    if source_vertex is None:
        source_vertex_idx = int(np.argmax(metric_values))
    else:
        source_vertex_idx = int(source_vertex)

    geodesic = _surface_geodesic_distances(verts, faces, source_vertex_idx)

    metric_source = float(metric_values[source_vertex_idx])
    if metric_source == 0.0:
        metric_range = float(np.max(metric_values) - np.min(metric_values))
        denom = metric_range if metric_range > 0 else 1.0
        rel_change = np.abs(metric_values - metric_source) / denom
    else:
        rel_change = np.abs(metric_values - metric_source) / abs(metric_source)

    order = np.argsort(geodesic)
    sorted_geo = geodesic[order]
    sorted_change = rel_change[order]

    decorrelation: dict[float, float] = {}
    for t in thresholds:
        meets = sorted_change >= t
        if meets.any():
            first_idx = int(np.argmax(meets))
            decorrelation[t] = float(sorted_geo[first_idx])
        else:
            decorrelation[t] = float("nan")

    rows = []
    for t in thresholds:
        rows.append(
            {
                "threshold": t,
                "decorrelation_um": decorrelation[t],
                "lesion_id": lesion_id,
                "n_surface_vertices": n_verts,
                "source_vertex_index": source_vertex_idx,
                "metric_at_source": metric_source,
            }
        )
    df = pd.DataFrame(rows)
    for t in thresholds:
        pct = int(round(t * 100))
        df[f"decorrelation_um_{pct}"] = decorrelation[t]

    _write_csv(df, csv_path)
    return df
