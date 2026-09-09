"""3D volume construction from registered 2D per-section outputs.

Assembles per-section tissue label maps (and optional nuclear feature
tables) into one ``SpatialData`` container: a ``"tissue_labels"``
``Labels3DModel`` under coordinate system ``"microns_3d"``.

Input: per-section ``(H, W)`` integer label maps at ``section_mpp``, already
warped/registered. Output: ``(z, y, x)`` uint8 volume at ``target_voxel_um``
isotropic, in microns. Z order follows manifest CSV row order, never
filename sorting.

Deliberate divergences: (1) rigid-only, elastic fields are not baked in;
(2) label z-pitch is always uniform ``target_voxel_um``; variable
``thickness_um`` applies to nuclear z only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from path3d.slide_io import load_manifest, load_manifest_thickness

if TYPE_CHECKING:
    from spatialdata.transformations import Affine


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
    target_voxel_um: float = 12.0,
    section_mpp: float = 0.5,
    level0_mpp: float | None = None,
    registration_mpp: float | None = None,
    output_path: str | Path | None = None,
    nuclear_parquet_path: str | Path | None = None,
    smooth_z_sigma: float = 1.0,
):
    """Assemble per-section label maps (+ optional nuclear features) into one
    3D SpatialData container.

    Section order/count come from ``load_manifest`` row order; z-index k =
    enumerate position k, label maps keyed by ``str(path)``. Label z-spacing
    is unconditionally ``target_voxel_um``, never ``thickness_um``. With
    ``nuclear_features`` (``transforms``, ``level0_mpp``,
    ``registration_mpp`` then required), centroids are mapped to microns via
    the section's rigid Affine, z overridden with cumulative-thickness z,
    and globally-unique instance IDs attached as ``tables["nuclei"]``.
    Output is rechunked to ``(1, 512, 512)``.

    Args:
        manifest_csv: manifest CSV (section_index, filename, path[, thickness_um]).
        label_maps: ``str(path)`` -> (H, W) integer label map, at ``section_mpp``.
        transforms: ``str(path)`` -> full rigid Affine. Required with ``nuclear_features``.
        nuclear_features: ``str(path)`` -> nuclear feature ``pd.DataFrame`` (level-0
            pixel centroids). Omit for no ``tables["nuclei"]``.
        target_voxel_um: isotropic output resolution, microns.
        section_mpp: input label-map resolution, microns per pixel.
        level0_mpp, registration_mpp: level-0 and Affine-fit resolutions, microns/px.
            Both required with ``nuclear_features``.
        output_path: if given, ``sdata.write()`` target.
        nuclear_parquet_path: if given, merged nuclear table written here.
        smooth_z_sigma: pre-mapping Affine in-plane smoothing sigma; ``0.0``
            disables. Unused without ``nuclear_features``.

    Returns:
        ``SpatialData`` with ``labels["tissue_labels"]``, plus ``tables["nuclei"]``
        when ``nuclear_features`` was given.

    Raises:
        ImportError: spatialdata (or anndata) is not installed.
        KeyError: a manifest path is missing from ``label_maps`` or ``nuclear_features``.
        ValueError: ``nuclear_features`` given without ``transforms``, ``level0_mpp``,
            or ``registration_mpp``.
    """
    try:
        from spatialdata import SpatialData
        from spatialdata.models import Labels3DModel
        from spatialdata.transformations import Affine
    except ImportError as exc:
        raise ImportError(
            "spatialdata is required. Install: pip install spatialdata"
        ) from exc

    paths = load_manifest(manifest_csv)
    n = len(paths)

    resized_sections: list[np.ndarray] = []
    for path in paths:
        key = str(path)
        if key not in label_maps:
            raise KeyError(
                f"No label map provided for manifest path: {key}. "
                f"label_maps must contain an entry for every manifest row."
            )
        resized_sections.append(
            _downsample_label_stack(label_maps[key], section_mpp, target_voxel_um)
        )

    vol = np.stack(resized_sections, axis=0).astype(np.uint8)

    # LABEL VOLUME Z IS ALWAYS UNIFORM (05-CONTEXT.md "Scope amendment" /
    # module docstring's "Deliberate scoping choice"): all-None input always
    # yields the uniform target_voxel_um pitch, regardless of any
    # thickness_um manifest column -- computed here purely to reuse the
    # shared helper and document the intent, not because thickness feeds in.
    _cumulative_z_from_thickness([None] * n, default=target_voxel_um)

    # Index (z_idx, y_idx, x_idx) -> microns (z_um, y_um, x_um). Isotropic:
    # every axis scales by target_voxel_um. The z-row's scale is ALWAYS
    # target_voxel_um -- it deliberately ignores thickness_um; see
    # 05-CONTEXT.md's "Scope amendment" for why a single linear Affine
    # cannot (and should not, per that decision) encode variable thickness.
    matrix = np.array([
        [target_voxel_um, 0.,               0.,               0.],  # z_um = target_voxel_um * z_idx (always uniform, ignores thickness_um by design)
        [0.,               target_voxel_um, 0.,               0.],  # y_um = target_voxel_um * y_idx
        [0.,               0.,               target_voxel_um, 0.],  # x_um = target_voxel_um * x_idx
        [0.,               0.,               0.,               1.],
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
        if transforms is None:
            raise ValueError(
                "transforms must be provided when nuclear_features is "
                "given -- nuclear centroids require the section's full "
                "rigid Affine to reach microns_3d (RESEARCH.md Pitfall 2: "
                "never the label's scale-only convention)."
            )
        if level0_mpp is None or registration_mpp is None:
            raise ValueError(
                "level0_mpp and registration_mpp must both be provided "
                "when nuclear_features is given -- required to rescale "
                "level-0 centroids onto the registration Affine's pixel "
                "grid."
            )

        # NUCLEAR z ONLY: cumulative-thickness placement from the optional
        # manifest thickness_um column (falls back to uniform
        # target_voxel_um when absent) -- 05-CONTEXT.md's "Scope amendment".
        # This is deliberately separate from the label z-affine above,
        # which is unconditionally uniform and never reads thickness_um.
        thickness_per_section = load_manifest_thickness(manifest_csv)
        z_um_per_section = _cumulative_z_from_thickness(
            thickness_per_section, default=target_voxel_um
        )

        # Smoothing (Plan 05-03 Task 3): rebuild each section's transform
        # from z-smoothed in-plane parameters when smooth_z_sigma > 0;
        # otherwise use transforms unchanged. Ordered by INTEGER section
        # index (enumerate(paths)), never by path string.
        working_transforms = _smoothed_section_transforms(
            paths, transforms, smooth_z_sigma
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
            centroids = feats[["centroid_y", "centroid_x"]].to_numpy(dtype=np.float64)
            zyx = _transform_centroids_to_microns(
                centroids, working_transforms[key], level0_mpp, registration_mpp
            )
            # NUCLEAR z OVERRIDE: discard the affine-derived z (which comes
            # from registration.py's own z convention and need not equal
            # the manifest thickness) and write the cumulative-thickness
            # value for this section instead. Keep only y, x from the
            # affine (see module docstring's "Deliberate scoping choice").
            zyx = zyx.copy()
            zyx[:, 0] = z_um_per_section[k]

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
