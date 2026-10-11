"""3D volume construction from registered 2D per-section outputs.

Assembles per-section tissue label maps (and optional nuclear feature
tables) into one ``SpatialData`` container: a ``"tissue_labels"``
``Labels3DModel`` under coordinate system ``"microns_3d"``.

Input: per-section ``(H, W)`` integer label maps at ``section_mpp``, already
warped/registered. Output: ``(z, y, x)`` uint8 volume of isotropic voxels, in
microns. Z order follows manifest CSV row order, never filename sorting.

Voxel size: given ``section_thickness_um`` and ``sampling_interval``, the
voxel is the physical spacing between imaged sections on every axis,
thickness x interval (4 um sections with every second one stained: 8 x 8 x 8
um), so each z-plane holds one section. Without them it is
``target_voxel_um`` (12 um by default).

Deliberate divergences: (1) nuclei use the full VALIS warp (rigid +
elastic) when their table carries ``reg_y_um``/``reg_x_um``, which
``pipeline.run_pipeline`` writes; tables without them fall back to the
section's rigid Affine, which omits the elastic field; (2) sections sit at
their physical z (manifest ``section_number`` x section spacing, or
cumulative ``thickness_um``), shared by labels and nuclei. The volume is a
uniform voxel grid, and a z-plane with no section within half a pitch (e.g.
where sections were removed) is an empty placeholder, listed in
``sdata.attrs["z_planes"]``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from path3d.slide_io import (
    load_manifest,
    load_manifest_section_numbers,
    load_manifest_thickness,
)

if TYPE_CHECKING:
    from spatialdata.transformations import Affine


# Nuclear-table columns holding centroids already warped into the registered
# canvas (microns), written by pipeline.run_pipeline.
_REGISTERED_COLUMNS = frozenset({"reg_y_um", "reg_x_um"})

# Voxel size when the section thickness and sampling interval are not given.
_DEFAULT_VOXEL_UM = 12.0


def _resolve_voxel_um(
    target_voxel_um: float | None,
    section_thickness_um: float | None,
    sampling_interval: int | None,
) -> tuple[float, float | None]:
    """Voxel size and section spacing, in microns.

    Consecutive imaged sections are ``section_thickness_um`` x
    ``sampling_interval`` apart (4 um sections, every second one stained:
    8 um), and the voxel is that spacing on every axis. Without either, the
    voxel is ``target_voxel_um`` (``_DEFAULT_VOXEL_UM`` when None) and the
    spacing is unknown (None).

    Raises:
        ValueError: only one of ``section_thickness_um``/``sampling_interval``
            is given, either is out of range, or ``target_voxel_um`` is also
            given and is not the spacing.
    """
    if section_thickness_um is None and sampling_interval is None:
        voxel_um = _DEFAULT_VOXEL_UM if target_voxel_um is None else float(target_voxel_um)
        return voxel_um, None
    if section_thickness_um is None or sampling_interval is None:
        raise ValueError(
            "Give section_thickness_um and sampling_interval together: sections "
            "are cut thickness x sampling interval apart (4 um sections with "
            "every 2nd one stained are 8 um apart)."
        )
    if section_thickness_um <= 0:
        raise ValueError(
            f"section_thickness_um must be > 0, got {section_thickness_um}"
        )
    if int(sampling_interval) != sampling_interval or sampling_interval < 1:
        raise ValueError(
            "sampling_interval must be a whole number >= 1 (1 = every section "
            f"stained, 2 = every second), got {sampling_interval}"
        )
    spacing_um = float(section_thickness_um) * int(sampling_interval)
    if target_voxel_um is not None and not np.isclose(target_voxel_um, spacing_um):
        raise ValueError(
            f"target_voxel_um = {target_voxel_um} um, but sections are "
            f"{section_thickness_um} um x every {int(sampling_interval)} = "
            f"{spacing_um} um apart, and the voxel is that spacing. Leave "
            f"target_voxel_um out."
        )
    return spacing_um, spacing_um


def _cumulative_z_from_thickness(
    thickness_per_section: list[float | None],
    default: float = 12.0,
) -> list[float]:
    """Cumulative z-placement from optional per-section thickness.

    z[i] = sum(thickness[0..i-1]); ``None`` falls back to ``default``.

    Args:
        thickness_per_section: per-section thickness in microns, or None.
        default: fallback thickness in microns when a value is None.

    Returns:
        Z-positions in microns, same length/order as the input.
    """
    z: list[float] = []
    running = 0.0
    for t in thickness_per_section:
        z.append(running)
        running += t if t is not None else default
    return z


def _section_z_um(
    section_numbers: list[int] | None,
    thickness_per_section: list[float | None],
    spacing_um: float,
) -> np.ndarray:
    """Physical z of each section in microns, the first section at 0.

    From ``section_numbers`` when the manifest has them (consecutive imaged
    sections are ``spacing_um`` apart, so a jump in numbers leaves a gap),
    else cumulative ``thickness_per_section`` with ``spacing_um`` for blanks.
    """
    if section_numbers is not None:
        numbers = np.asarray(section_numbers, dtype=np.float64)
        return (numbers - numbers[0]) * spacing_um
    return np.asarray(
        _cumulative_z_from_thickness(thickness_per_section, default=spacing_um),
        dtype=np.float64,
    )


def _plane_sources(section_z_um: np.ndarray, target_voxel_um: float) -> np.ndarray:
    """Which section each plane of the ``target_voxel_um`` z grid shows.

    A plane shows the nearest section when it lies within half a pitch (the
    larger of the closest section spacing and the voxel size). Otherwise no
    section was imaged there and the plane is a placeholder (-1). Nearest-
    section lookup keeps the labels categorical.

    Returns:
        (n_planes,) int array of manifest positions, -1 for placeholders.
    """
    n_planes = int(np.floor(section_z_um[-1] / target_voxel_um + 1e-6)) + 1
    plane_z = np.arange(n_planes) * target_voxel_um
    nearest = np.abs(plane_z[:, None] - section_z_um[None, :]).argmin(axis=1)
    steps = np.diff(section_z_um)
    pitch = max(target_voxel_um, float(steps.min()) if steps.size else 0.0)
    too_far = np.abs(plane_z - section_z_um[nearest]) > pitch / 2 + 1e-6
    return np.where(too_far, -1, nearest)


def _downsample_label_stack(
    label_map: np.ndarray,
    section_mpp: float,
    target_voxel_um: float,
) -> np.ndarray:
    """Resize a single integer label map to ``target_voxel_um`` in-plane resolution.

    Always nearest-neighbour (order=0) -- label maps are categorical and
    must never be bilinear/bicubic-resized.

    Args:
        label_map: (H, W) integer-dtype categorical label array.
        section_mpp: input resolution, microns per pixel.
        target_voxel_um: output resolution, microns per pixel (isotropic).

    Returns:
        (H', W'), same dtype as ``label_map``, values already present.

    Raises:
        ValueError: ``label_map`` is not an integer dtype.
    """
    if label_map.dtype.kind not in "iu":
        raise ValueError(
            f"label_map must have an integer dtype for nearest-neighbor "
            f"resizing, got {label_map.dtype}. Never float-cast label maps "
            f"before resizing."
        )

    from skimage.transform import resize

    scale = section_mpp / target_voxel_um
    h, w = label_map.shape
    out_h = max(1, round(h * scale))
    out_w = max(1, round(w * scale))

    resized = resize(
        label_map,
        (out_h, out_w),
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    )
    return resized.astype(label_map.dtype)


def _smoothed_section_transforms(
    paths: list[Path],
    transforms: dict[str, Affine],
    smooth_z_sigma: float,
) -> dict[str, Affine]:
    """Return per-path Affines with in-plane parameters Gaussian-smoothed along z.

    When ``smooth_z_sigma <= 0``, ``transforms`` is returned unchanged.
    Sections are ordered by integer section index (``enumerate(paths)``),
    never by path string. The rebuilt Affine's z row is a placeholder --
    the caller overrides it. Smooths transform parameters only, never
    label pixels.
    """
    if smooth_z_sigma <= 0:
        return transforms

    from spatialdata.transformations import Affine

    keys = [str(path) for path in paths]
    params = np.stack(
        [
            transforms[key]
            .to_affine_matrix(input_axes=("y", "x"), output_axes=("y", "x"))[:2] 
            .ravel()
            for key in keys
        ],
        axis=0,
    )
    smoothed = _smooth_transforms_z(params, sigma=smooth_z_sigma)

    rebuilt: dict[str, Affine] = {}
    for key, row in zip(keys, smoothed):
        a00, a01, a02, a10, a11, a12 = row
        matrix = np.array(
            [
                [0.0, 0.0, 0.0],  # z row -- placeholder, overridden by caller
                [a00, a01, a02],  # y_out
                [a10, a11, a12],  # x_out
                [0.0, 0.0, 1.0],
            ]
        )
        rebuilt[key] = Affine(
            matrix, input_axes=("y", "x"), output_axes=("z", "y", "x")
        )
    return rebuilt


def build_volume(
    manifest_csv: str | Path,
    label_maps: dict[str, np.ndarray],
    *,
    transforms: dict[str, Affine] | None = None,
    nuclear_features: dict[str, pd.DataFrame] | None = None,
    target_voxel_um: float | None = None,
    section_mpp: float = 0.5,
    level0_mpp: float | None = None,
    registration_mpp: float | None = None,
    output_path: str | Path | None = None,
    nuclear_parquet_path: str | Path | None = None,
    smooth_z_sigma: float = 0.0,
    section_thickness_um: float | None = None,
    sampling_interval: int | None = None,
):
    """Assemble per-section label maps (+ optional nuclear features) into one
    3D SpatialData container.

    Section order/count come from ``load_manifest`` row order, label maps
    keyed by ``str(path)``. Consecutive imaged sections are
    ``section_thickness_um`` x ``sampling_interval`` apart, and the voxel is
    that spacing on every axis (8 x 8 x 8 um for 4 um sections with every
    second one stained). Each section sits at its physical z: the manifest's
    ``section_number`` column times the spacing when present (a jump in
    numbers is a gap), else cumulative ``thickness_um``, else the spacing
    apart. The label volume is a uniform voxel grid starting at the first
    section; each z-plane shows the nearest section within half a pitch, and
    planes with none (a gap) are left empty (0). ``sdata.attrs["z_planes"]``
    records each plane's ``z_um`` and ``section_index`` (manifest position,
    -1 for a placeholder), and ``sdata.attrs["voxel"]`` the voxel size and
    the thickness and interval it came from. With
    ``nuclear_features``, each nucleus's in-plane position comes from its
    table's ``reg_y_um``/``reg_x_um`` columns when present: centroids already
    mapped through VALIS's rigid + elastic point warp
    (``registration.warp_points_to_registered``), i.e. the transform the label
    maps were warped with. Tables without those columns fall back to the
    section's rigid Affine (``transforms``, ``level0_mpp`` and
    ``registration_mpp`` then required), which omits the elastic field.
    Globally-unique instance IDs are attached as ``tables["nuclei"]``; nuclei take their section's
    physical z, the same z its labels are placed at. Output is rechunked to
    ``(1, 512, 512)``.

    Args:
        manifest_csv: manifest CSV (section_index, filename, path[,
            thickness_um][, section_number]). ``section_number`` counts
            imaged sections (consecutive stained sections differ by 1), and
            ``thickness_um`` is the distance to the next manifest row; with
            ``section_thickness_um`` it must equal the section spacing.
        label_maps: ``str(path)`` -> (H, W) integer label map, at ``section_mpp``.
        transforms: ``str(path)`` -> full rigid Affine. Required when a
            ``nuclear_features`` table lacks ``reg_y_um``/``reg_x_um``.
        nuclear_features: ``str(path)`` -> nuclear feature ``pd.DataFrame``
            (``reg_y_um``/``reg_x_um`` in the registered canvas, or level-0
            pixel centroids). Omit for no ``tables["nuclei"]``.
        target_voxel_um: isotropic output resolution, microns, used only
            without ``section_thickness_um``/``sampling_interval``. None means
            12 um. Passing a value other than the section spacing raises.
        section_mpp: input label-map resolution, microns per pixel.
        level0_mpp, registration_mpp: level-0 and Affine-fit resolutions, microns/px.
            Required with ``transforms``.
        output_path: if given, ``sdata.write()`` target.
        nuclear_parquet_path: if given, merged nuclear table written here.
        smooth_z_sigma: in-plane smoothing sigma (in sections) applied to the
            rigid Affines before mapping. Default ``0.0`` (off): adjacent
            sections sit on their slides at very different angles, so
            averaging their parameters gives transforms that belong to no
            section, and the label maps are never smoothed. Only used on the
            rigid-Affine fallback.
        section_thickness_um: thickness each section was cut at, microns.
            Give with ``sampling_interval``.
        sampling_interval: every how many cut sections one was stained and
            imaged (1 = every section, 2 = every second). With
            ``section_thickness_um`` it sets the section spacing and the
            voxel size. Without both, consecutive sections are placed
            ``target_voxel_um`` apart with a warning, unless every manifest
            row has a ``thickness_um``.

    Returns:
        ``SpatialData`` with ``labels["tissue_labels"]``, plus ``tables["nuclei"]``
        when ``nuclear_features`` was given.

    Raises:
        ImportError: spatialdata (or anndata) is not installed.
        KeyError: a manifest path is missing from ``label_maps`` or ``nuclear_features``.
        ValueError: a ``nuclear_features`` table lacks ``reg_y_um``/``reg_x_um``
            and ``transforms``, ``level0_mpp`` or ``registration_mpp`` is missing;
            only one of ``section_thickness_um``/``sampling_interval`` is
            given; ``target_voxel_um`` or the manifest's ``thickness_um``
            disagrees with the section spacing.
    """
    try:
        from spatialdata import SpatialData
        from spatialdata.models import Labels3DModel
        from spatialdata.transformations import Affine
    except ImportError as exc:
        raise ImportError(
            "spatialdata is required. Install: pip install spatialdata"
        ) from exc

    voxel_um, spacing_um = _resolve_voxel_um(
        target_voxel_um, section_thickness_um, sampling_interval
    )
    paths = load_manifest(manifest_csv)

    resized_sections: list[np.ndarray] = []
    for path in paths:
        key = str(path)
        if key not in label_maps:
            raise KeyError(
                f"No label map provided for manifest path: {key}. "
                f"label_maps must contain an entry for every manifest row."
            )
        resized_sections.append(
            _downsample_label_stack(label_maps[key], section_mpp, voxel_um)
        )

    stack = np.stack(resized_sections, axis=0).astype(np.uint8)

    # Physical z of every section; labels and nuclei both use it, so a
    # nucleus is always looked up in its own section.
    section_numbers = load_manifest_section_numbers(manifest_csv)
    thickness_per_section = load_manifest_thickness(manifest_csv)
    if spacing_um is None:
        if section_numbers is not None or None in thickness_per_section:
            warnings.warn(
                f"section_thickness_um and sampling_interval not given; placing "
                f"consecutive sections {voxel_um} um apart, the voxel size. Pass "
                f"both to place sections at their physical z.",
                stacklevel=2,
            )
        spacing_um = voxel_um
    else:
        listed = sorted({t for t in thickness_per_section if t is not None})
        if any(not np.isclose(t, spacing_um) for t in listed):
            raise ValueError(
                f"Manifest thickness_um {listed} disagrees with the section "
                f"spacing, {section_thickness_um} um x every "
                f"{int(sampling_interval)} = {spacing_um} um. The column is the "
                f"distance between consecutive manifest rows; mark removed "
                f"sections with a section_number column instead."
            )
    section_z_um = _section_z_um(section_numbers, thickness_per_section, spacing_um)

    # Place sections on the uniform z grid; gaps become empty placeholders
    # rather than being closed up.
    sources = _plane_sources(section_z_um, voxel_um)
    has_section = sources >= 0
    vol = np.zeros((len(sources), *stack.shape[1:]), dtype=np.uint8)
    vol[has_section] = stack[sources[has_section]]
    if not has_section.all():
        empty_z = (np.flatnonzero(~has_section) * voxel_um).tolist()
        warnings.warn(
            f"{len(empty_z)} of {len(sources)} z-planes have no section within "
            f"half a pitch and are left empty (z = {empty_z[:10]} um"
            f"{' ...' if len(empty_z) > 10 else ''}); see sdata.attrs['z_planes'].",
            stacklevel=2,
        )

    # Index (z_idx, y_idx, x_idx) -> microns (z_um, y_um, x_um). Isotropic:
    # every axis scales by voxel_um; sections were placed on this z grid at
    # their physical z above.
    matrix = np.array([
        [voxel_um, 0.,       0.,       0.],  # z_um = voxel_um * z_idx
        [0.,       voxel_um, 0.,       0.],  # y_um = voxel_um * y_idx
        [0.,       0.,       voxel_um, 0.],  # x_um = voxel_um * x_idx
        [0.,       0.,       0.,       1.],
    ])
    affine = Affine(matrix, input_axes=("z", "y", "x"), output_axes=("z", "y", "x"))

    labels_el = Labels3DModel.parse(
        vol,
        dims=("z", "y", "x"),
        transformations={"microns_3d": affine},
    )
    # chunks= kwarg to .parse() is silently ignored for numpy input
    # (RESEARCH.md Pitfall 1) -- always rechunk explicitly after parsing.
    labels_el = labels_el.chunk({"z": 1, "y": 512, "x": 512})

    tables: dict[str, object] = {}

    if nuclear_features is not None:
        needs_affine = any(
            not _REGISTERED_COLUMNS.issubset(df.columns)
            for df in nuclear_features.values()
        )
        if needs_affine and transforms is None:
            raise ValueError(
                "transforms must be provided when nuclear_features is "
                "given -- nuclear centroids require the section's full "
                "rigid Affine to reach microns_3d (RESEARCH.md Pitfall 2: "
                "never the label's scale-only convention)."
            )
        if needs_affine and (level0_mpp is None or registration_mpp is None):
            raise ValueError(
                "level0_mpp and registration_mpp must both be provided "
                "when nuclear_features is given -- required to rescale "
                "level-0 centroids onto the registration Affine's pixel "
                "grid."
            )

        # Smoothing (Plan 05-03 Task 3): rebuild each section's transform
        # from z-smoothed in-plane parameters when smooth_z_sigma > 0;
        # otherwise use transforms unchanged. Ordered by INTEGER section
        # index (enumerate(paths)), never by path string.
        working_transforms = (
            _smoothed_section_transforms(paths, transforms, smooth_z_sigma)
            if needs_affine else {}
        )

        per_section_frames: dict[int, pd.DataFrame] = {}
        for k, path in enumerate(paths):
            key = str(path)
            if key not in nuclear_features:
                raise KeyError(
                    f"No nuclear feature table provided for manifest path: "
                    f"{key}. nuclear_features must contain an entry for "
                    f"every manifest row when supplied."
                )
            feats = nuclear_features[key]
            if _REGISTERED_COLUMNS.issubset(feats.columns):
                # Already in the registered canvas, through the same rigid +
                # elastic warp as the label maps.
                reg_yx = feats[["reg_y_um", "reg_x_um"]].to_numpy(dtype=np.float64)
                zyx = np.column_stack([np.zeros(len(reg_yx)), reg_yx])
            else:
                # Rigid-only fallback: misses the elastic field the label maps
                # were warped with, so nuclei can sit off their labels by the
                # local elastic displacement.
                centroids = feats[["centroid_y", "centroid_x"]].to_numpy(dtype=np.float64)
                zyx = _transform_centroids_to_microns(
                    centroids, working_transforms[key], level0_mpp, registration_mpp
                )
            # NUCLEAR z OVERRIDE: discard the affine-derived z (which comes
            # from registration.py's own z convention) and use the section's
            # physical z, the same z its label map was placed at.
            zyx = zyx.copy()
            zyx[:, 0] = section_z_um[k]

            df = feats.copy()
            df["z_um"] = zyx[:, 0]
            df["y_um"] = zyx[:, 1]
            df["x_um"] = zyx[:, 2]
            per_section_frames[k] = df

        merged = _assign_global_instance_ids(per_section_frames)
        # Single shared region for every nuclei row -- RESEARCH.md Pitfall 5
        # (never per-section keys like the label pathway's "seg_{idx:04d}").
        merged["region"] = "tissue_labels"

        if nuclear_parquet_path is not None:
            merged.to_parquet(str(nuclear_parquet_path), engine="pyarrow")

        try:
            import anndata as ad
            from spatialdata.models import TableModel
        except ImportError as exc:
            raise ImportError(
                "anndata and spatialdata are required. "
                "Install: pip install anndata spatialdata"
            ) from exc

        spatial_3d = merged[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
        obs = merged.drop(columns=["z_um", "y_um", "x_um"]).reset_index(drop=True)
        obs.index = obs.index.astype(str)
        adata = ad.AnnData(obs=obs)
        adata.obsm["spatial_3d"] = spatial_3d

        tables["nuclei"] = TableModel.parse(
            adata,
            region="tissue_labels",
            region_key="region",
            instance_key="instance_id",
        )

    sdata = SpatialData(labels={"tissue_labels": labels_el}, tables=tables)
    sdata.attrs["z_planes"] = {
        "z_um": (np.arange(len(sources)) * voxel_um).tolist(),
        "section_index": [int(i) for i in sources],
    }
    sdata.attrs["voxel"] = {
        "voxel_um": voxel_um,
        "section_thickness_um": section_thickness_um,
        "sampling_interval": None if sampling_interval is None else int(sampling_interval),
    }

    if output_path is not None:
        sdata.write(str(output_path))

    return sdata


# ---------------------------------------------------------------------------
# Plan 05-03 (VOL-02): nuclear pathway helpers
# ---------------------------------------------------------------------------


def _transform_centroids_to_microns(
    centroids_level0_yx: np.ndarray,
    affine: Affine,
    level0_mpp: float,
    registration_mpp: float,
) -> np.ndarray:
    """Map level-0-pixel nuclear centroids to microns_3d via a rigid Affine.

    Builds the matrix via ``to_affine_matrix`` and matmuls explicitly, since
    ``Affine`` has no point-array transform. Needs the FULL rigid Affine
    (nuclei come from unwarped slides), unlike label maps.

    Args:
        centroids_level0_yx: (N, 2) [y_px, x_px] at level-0 resolution.
        affine: full rigid Affine (input_axes=("y","x"), output_axes=("z","y","x")).
        level0_mpp, registration_mpp: level-0 and Affine-fit resolutions, microns/px.

    Returns:
        (N, 3) float64 [z_um, y_um, x_um]; ``build_volume`` overrides z.
    """
    scale = level0_mpp / registration_mpp
    pts_reg_mpp = np.asarray(centroids_level0_yx, dtype=np.float64) * scale
    matrix = affine.to_affine_matrix(input_axes=("y", "x"), output_axes=("z", "y", "x"))
    n = pts_reg_mpp.shape[0]
    homog = np.concatenate([pts_reg_mpp, np.ones((n, 1))], axis=1)
    out = (matrix @ homog.T).T
    return out[:, :3]


def _assign_global_instance_ids(
    per_section_features: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Offset each section's local instance IDs into one globally-unique column.

    ``nuclear_detection.py`` restarts its instance-ID counter per section;
    keys here must be int section indices, iterated via ``sorted(...)``.

    Args:
        per_section_features: int section index -> feature ``pd.DataFrame``
            (``section_index``/``instance_id`` added/overwritten).

    Returns:
        Concatenated ``pd.DataFrame``, row order = ascending section index
        then within-section order, with globally-unique ``instance_id``.

    Raises:
        TypeError: any key in ``per_section_features`` is not an ``int``.
    """
    if not all(isinstance(k, int) for k in per_section_features):
        raise TypeError(
            "per_section_features keys must be int section indices, not "
            "path strings -- re-key by enumerate(paths) before calling "
            "_assign_global_instance_ids (checker Warning 1)."
        )

    offset = 0
    frames: list[pd.DataFrame] = []
    for idx in sorted(per_section_features):
        df = per_section_features[idx].copy()
        df["instance_id"] = np.arange(1, len(df) + 1) + offset
        df["section_index"] = idx
        offset += len(df)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _smooth_transforms_z(affine_params: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gently smooth per-section rigid-transform parameters along z.

    Rows ``[a00, a01, a02, a10, a11, a12]`` ordered by section index; 1D
    Gaussian along axis 0, ``mode="nearest"``. Parameters only, never
    label pixels; valid only for small inter-section rotation deltas.

    Args:
        affine_params: (N_sections, 6) per-section in-plane affine params.
        sigma: Gaussian smoothing sigma (in sections).

    Returns:
        Smoothed array, same shape.
    """
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(affine_params, sigma=sigma, axis=0, mode="nearest")


def fill_missing_z(sdata, method: str = "nearest"):
    """Forward-compatible hook for missing-z-slice interpolation.

    Only ``"nearest"`` is implemented, as a pass-through no-op.

    Args:
        sdata: a spatialdata.SpatialData container.
        method: ``"nearest"`` (implemented); ``"linear"``/``"optical_flow"``
            or anything else raise (see Raises).

    Returns:
        ``sdata``, unchanged.

    Raises:
        NotImplementedError: ``method`` is ``"linear"`` or ``"optical_flow"``.
        ValueError: ``method`` is not a recognized value.
    """
    if method == "nearest":
        return sdata
    if method == "linear":
        raise NotImplementedError(
            "linear z-fill is deferred to a later tier -- see 05-RESEARCH "
            "Open Question 3"
        )
    if method == "optical_flow":
        raise NotImplementedError("optical_flow z-fill is Tier 3 -- not implemented")
    raise ValueError(f"unknown method: {method}")
