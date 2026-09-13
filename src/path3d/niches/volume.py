"""Assemble per-section niche rasters into a 3D SpatialData volume.

Produces **two** elements from the same stack:

``images["niche_probabilities"]``
    ``(c, z, y, x)`` float32 ``Image3DModel``, one channel per class, NaN
    off-tissue. This is the quantitative object -- the model card asks you to
    "use the continuous ``p_*`` columns wherever you can".

``labels["tissue_labels"]``
    ``(z, y, x)`` uint8 ``Labels3DModel`` from the per-voxel argmax. Built
    because ``visualization.view_volume`` and ``quantification`` need integer
    labels, and named ``tissue_labels`` so both work on it unchanged (pass
    ``tissue_type="HGSC_niches"``). It inherits the argmax bias the model card
    warns about -- prefer the probability volume for any number you report.

Two deliberate divergences from :func:`path3d.volume.build_volume`:

1. **Voxels are anisotropic by default.** ``build_volume`` forces an isotropic
   ``target_voxel_um`` that doubles as the z pitch. For niches that is a trap:
   the in-plane grid is 32 um and sections are typically ~12 um apart, so
   isotropy means either upsampling in-plane 2.7x (7x the memory, no new
   information) or stretching z by 2.7x (wrong geometry, wrong shape
   descriptors). This module keeps the native ``(z_spacing_um, grid_um,
   grid_um)`` and encodes the anisotropy in the affine, where it belongs.
   Pass ``target_voxel_um`` to force isotropy when a downstream consumer needs
   it -- ``quantification.compute_volumetrics`` takes a single scalar
   ``voxel_um`` and assumes cubic voxels.
2. **z spacing is explicit.** ``build_volume`` derives label z from
   ``target_voxel_um`` and ignores the manifest's ``thickness_um``. Here
   ``z_spacing_um`` is a required argument (or read from a uniform manifest
   ``thickness_um`` column), because getting it wrong scales every volume and
   surface area you subsequently compute.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from path3d.config import NICHE_BACKGROUND_INDEX
from path3d.niches.rasterize import NicheSection, argmax_labels, smooth_probs
from path3d.slide_io import load_manifest, load_manifest_thickness

if TYPE_CHECKING:
    from spatialdata import SpatialData

# Warn past this many voxels per channel -- a float32 probability volume is
# 4 bytes x n_classes x this, and it is built in RAM before being chunked.
_VOXEL_WARN_THRESHOLD = 200_000_000


def resolve_z_spacing(
    manifest_csv: str | Path, z_spacing_um: float | None
) -> float:
    """Settle the inter-section z spacing, in microns.

    Args:
        manifest_csv: manifest with an optional ``thickness_um`` column.
        z_spacing_um: explicit spacing; when None, read from the manifest.

    Returns:
        The spacing to use.

    Raises:
        ValueError: ``z_spacing_um`` is None and the manifest has no usable
            uniform ``thickness_um``, or the value is not positive.

    Note:
        This is the spacing between **consecutive manifest rows**, which is
        section thickness x sampling interval. If you cut 4 um sections and
        imaged every third one, it is 12 um, not 4 um.
    """
    if z_spacing_um is not None:
        if z_spacing_um <= 0:
            raise ValueError(f"z_spacing_um must be > 0, got {z_spacing_um}")
        return float(z_spacing_um)

    thickness = [t for t in load_manifest_thickness(manifest_csv) if t is not None]
    if not thickness:
        raise ValueError(
            "z_spacing_um was not given and the manifest has no thickness_um "
            "column. Pass z_spacing_um explicitly -- it is the spacing between "
            "CONSECUTIVE MANIFEST ROWS (section thickness x sampling "
            "interval), and getting it wrong rescales every volume and "
            "surface area computed from this stack."
        )
    unique = sorted({round(float(t), 6) for t in thickness})
    if len(unique) > 1:
        raise ValueError(
            f"manifest thickness_um is not uniform ({unique}), and a single "
            f"affine cannot encode variable z spacing. Pass z_spacing_um "
            f"explicitly to choose one nominal value."
        )
    return float(unique[0])


def _resample_inplane(
    array: np.ndarray, from_um: float, to_um: float, *, is_label: bool
) -> np.ndarray:
    """Nearest-neighbour in-plane resample of the trailing two axes.

    Nearest for BOTH probabilities and labels, deliberately. This path only
    ever upsamples (see the caller's guard), and upsampling adds no
    information -- replicating cells is honest, while bilinear would
    manufacture intermediate probabilities and smear the NaN off-tissue mask
    into the tissue.
    """
    if from_um == to_um:
        return array

    from skimage.transform import resize

    scale = from_um / to_um
    *lead, h, w = array.shape
    out_hw = (max(1, round(h * scale)), max(1, round(w * scale)))
    out = np.empty((*lead, *out_hw), array.dtype)

    flat_in = array.reshape(-1, h, w)
    flat_out = out.reshape(-1, *out_hw)
    for i in range(flat_in.shape[0]):
        flat_out[i] = resize(
            flat_in[i], out_hw, order=0, anti_aliasing=False, preserve_range=True
        ).astype(array.dtype)
    return out


def build_niche_volume(
    manifest_csv: str | Path,
    sections: dict[str, NicheSection],
    *,
    z_spacing_um: float | None = None,
    smooth_um: float = 0.0,
    smooth_z_sigma: float = 0.0,
    target_voxel_um: float | None = None,
    labels_key: str = "tissue_labels",
    probs_key: str = "niche_probabilities",
    output_path: str | Path | None = None,
    coord_system: str = "microns_3d",
) -> "SpatialData":
    """Stack per-section niche rasters into a 3D SpatialData container.

    Section order and count come from ``load_manifest`` row order -- z index k
    is manifest row k -- and ``sections`` is keyed by ``str(path)``, matching
    :func:`path3d.volume.build_volume`'s convention.

    Args:
        manifest_csv: manifest CSV (``section_index, filename, path``, with an
            optional ``thickness_um`` column).
        sections: ``str(path)`` -> :class:`NicheSection`, one per manifest row.
            All must share a grid shape, class list and ``grid_um``.
        z_spacing_um: microns between consecutive manifest rows. Read from a
            uniform manifest ``thickness_um`` when None. See
            :func:`resolve_z_spacing`.
        smooth_um: in-plane Gaussian sigma in **microns**, applied to
            probabilities before the argmax. The continuous form of the model
            card's 250 um reporting recommendation; ``0`` disables. A sigma
            around ``100`` gives roughly the noise reduction of 250 um
            reporting while keeping the native grid.
        smooth_z_sigma: Gaussian sigma **in sections**, applied across z after
            in-plane smoothing. ``0`` (the default) disables it -- enable only
            if you trust the registration to better than one grid cell, since
            it mixes neighbouring sections' probabilities.
        target_voxel_um: force isotropic voxels of this size, resampling
            in-plane. Leave None to keep native anisotropic
            ``(z_spacing_um, grid_um, grid_um)``. Required if you intend to
            call ``quantification.compute_volumetrics``, which assumes cubic
            voxels.
        labels_key: element name for the argmax label volume. Defaults to
            ``"tissue_labels"`` so ``view_volume`` and ``quantification`` find
            it without arguments.
        probs_key: element name for the probability volume.
        output_path: when given, ``sdata.write()`` target.
        coord_system: coordinate system name for both elements.

    Returns:
        ``SpatialData`` with ``images[probs_key]`` and ``labels[labels_key]``.

    Raises:
        ImportError: spatialdata is not installed (``pip install path3d[volume]``).
        KeyError: a manifest path has no entry in ``sections``.
        ValueError: sections disagree on grid shape, classes or ``grid_um``;
            or ``target_voxel_um`` would downsample (rasterise at a coarser
            ``grid_um`` instead, which averages properly rather than aliasing).
    """
    try:
        from spatialdata import SpatialData
        from spatialdata.models import Image3DModel, Labels3DModel
        from spatialdata.transformations import Affine
    except ImportError as exc:
        raise ImportError(
            "spatialdata is required. Install: pip install path3d[volume]"
        ) from exc

    paths = load_manifest(manifest_csv)
    z_um = resolve_z_spacing(manifest_csv, z_spacing_um)

    ordered: list[NicheSection] = []
    for path in paths:
        key = str(path)
        if key not in sections:
            raise KeyError(
                f"No niche raster provided for manifest path: {key}. sections "
                f"must contain an entry for every manifest row."
            )
        ordered.append(sections[key])

    reference = ordered[0]
    classes = list(reference.classes)
    grid_um = reference.grid_um
    for path, section in zip(paths, ordered):
        if section.shape != reference.shape:
            raise ValueError(
                f"Section {path.name} has grid shape {section.shape}, expected "
                f"{reference.shape}. Every section's grid must match, which "
                f"means every section must be rasterised on the SAME canvas -- "
                f"use the registered OME-TIFFs from one VALIS run (they share "
                f"an aligned canvas), not the raw slides."
            )
        if list(section.classes) != classes:
            raise ValueError(
                f"Section {path.name} has classes {section.classes}, expected "
                f"{classes}."
            )
        if section.grid_um != grid_um:
            raise ValueError(
                f"Section {path.name} has grid_um {section.grid_um}, expected "
                f"{grid_um}."
            )

    n_voxels = len(ordered) * int(np.prod(reference.shape))
    if n_voxels > _VOXEL_WARN_THRESHOLD:
        warnings.warn(
            f"Building a {len(ordered)} x {reference.shape[0]} x "
            f"{reference.shape[1]} volume ({n_voxels:,} voxels); the float32 "
            f"probability stack alone is "
            f"~{n_voxels * 4 * len(classes) / 1e9:.1f} GB in RAM. Consider a "
            f"coarser grid_um when rasterising.",
            stacklevel=2,
        )

    # (c, z, y, x)
    probs = np.stack([s.probs for s in ordered], axis=1).astype(np.float32)

    if smooth_um > 0:
        sigma_cells = smooth_um / grid_um
        probs = np.stack(
            [smooth_probs(probs[:, k], sigma_cells) for k in range(probs.shape[1])],
            axis=1,
        )

    if smooth_z_sigma > 0:
        from scipy.ndimage import gaussian_filter1d

        valid = np.isfinite(probs)
        filled = np.nan_to_num(probs, nan=0.0)
        num = gaussian_filter1d(filled, smooth_z_sigma, axis=1, mode="nearest")
        den = gaussian_filter1d(
            valid.astype(np.float32), smooth_z_sigma, axis=1, mode="nearest"
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            probs = np.where(den > 1e-8, num / den, np.nan).astype(np.float32)
        # A voxel with no prediction stays unpredicted -- z-smoothing must not
        # invent tissue in the gaps between sections.
        probs[~valid] = np.nan

    labels = np.stack(
        [argmax_labels(probs[:, k], classes) for k in range(probs.shape[1])], axis=0
    ).astype(np.uint8)

    voxel_zyx = (z_um, grid_um, grid_um)
    if target_voxel_um is not None:
        if target_voxel_um <= 0:
            raise ValueError(
                f"target_voxel_um must be > 0, got {target_voxel_um}"
            )
        if target_voxel_um > grid_um:
            raise ValueError(
                f"target_voxel_um ({target_voxel_um}) is coarser than the niche "
                f"grid ({grid_um} um); that would alias. Rasterise at "
                f"grid_um={target_voxel_um} instead -- rasterize_tiles averages "
                f"the tiles in each cell, which is the correct way to coarsen."
            )
        probs = _resample_inplane(probs, grid_um, target_voxel_um, is_label=False)
        labels = _resample_inplane(
            labels, grid_um, target_voxel_um, is_label=True
        )
        voxel_zyx = (z_um, target_voxel_um, target_voxel_um)

    # Index (z, y, x) -> microns. Diagonal, but NOT uniform: the z scale is the
    # physical inter-section spacing and the in-plane scales are the niche grid
    # size. Writing them independently is the whole point of this module.
    matrix = np.array(
        [
            [voxel_zyx[0], 0.0, 0.0, 0.0],
            [0.0, voxel_zyx[1], 0.0, 0.0],
            [0.0, 0.0, voxel_zyx[2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    affine = Affine(
        matrix, input_axes=("z", "y", "x"), output_axes=("z", "y", "x")
    )

    probs_el = Image3DModel.parse(
        probs,
        dims=("c", "z", "y", "x"),
        c_coords=list(classes),
        transformations={coord_system: affine},
    )
    labels_el = Labels3DModel.parse(
        labels, dims=("z", "y", "x"), transformations={coord_system: affine}
    )
    # chunks= on .parse() is silently ignored for numpy input -- always
    # rechunk explicitly afterwards (same pitfall as volume.build_volume).
    probs_el = probs_el.chunk({"c": len(classes), "z": 1, "y": 512, "x": 512})
    labels_el = labels_el.chunk({"z": 1, "y": 512, "x": 512})

    sdata = SpatialData(
        images={probs_key: probs_el}, labels={labels_key: labels_el}
    )
    sdata.attrs["niche_classes"] = list(classes)
    sdata.attrs["niche_voxel_um_zyx"] = [float(v) for v in voxel_zyx]
    sdata.attrs["niche_background_index"] = int(NICHE_BACKGROUND_INDEX)

    if output_path is not None:
        sdata.write(str(output_path))

    return sdata
