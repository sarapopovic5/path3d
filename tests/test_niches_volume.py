"""Tests for 3D niche volume assembly.

The properties that matter are geometric: z must be the physical inter-section
spacing (not the in-plane grid size), the two elements must agree on that
geometry, and a shape mismatch between sections must fail loudly rather than
produce a silently sheared volume.
"""

from __future__ import annotations

import numpy as np
import pytest

from path3d.config import NICHE_LABEL_INDEX
from path3d.niches.rasterize import NicheSection
from path3d.niches.volume import build_niche_volume, resolve_z_spacing

pytest.importorskip("spatialdata")

CLASSES = ["acellular", "epithelium", "immune", "stroma"]


def _section(shape=(8, 10), winner="epithelium", grid_um=32.0):
    """A NicheSection where `winner` dominates everywhere on tissue."""
    probs = np.full((len(CLASSES), *shape), 0.1, np.float32)
    probs[CLASSES.index(winner)] = 0.7
    return NicheSection(
        probs=probs,
        classes=list(CLASSES),
        tile_count=np.ones(shape, np.int32),
        grid_um=grid_um,
        canvas_wh=(shape[1] * 64, shape[0] * 64),
        mpp=0.5,
    )


def _manifest(tmp_path, n, thickness=None):
    rows = ["section_index,filename,path" + (",thickness_um" if thickness else "")]
    paths = []
    for i in range(n):
        p = tmp_path / f"{i:04d}.ome.tiff"
        p.touch()
        paths.append(p)
        row = f"{i},{p.name},{p}"
        if thickness:
            row += f",{thickness[i]}"
        rows.append(row)
    csv = tmp_path / "manifest.csv"
    csv.write_text("\n".join(rows) + "\n")
    return csv, paths


class TestZSpacing:
    def test_explicit_spacing_wins(self, tmp_path):
        csv, _ = _manifest(tmp_path, 2, thickness=[4.0, 4.0])
        assert resolve_z_spacing(csv, 12.0) == 12.0

    def test_uniform_manifest_thickness_is_used(self, tmp_path):
        csv, _ = _manifest(tmp_path, 3, thickness=[12.0, 12.0, 12.0])
        assert resolve_z_spacing(csv, None) == 12.0

    def test_missing_thickness_raises_rather_than_guessing(self, tmp_path):
        csv, _ = _manifest(tmp_path, 2)
        with pytest.raises(ValueError, match="CONSECUTIVE MANIFEST ROWS"):
            resolve_z_spacing(csv, None)

    def test_non_uniform_thickness_raises(self, tmp_path):
        csv, _ = _manifest(tmp_path, 3, thickness=[4.0, 8.0, 4.0])
        with pytest.raises(ValueError, match="not uniform"):
            resolve_z_spacing(csv, None)


class TestVolumeGeometry:
    def test_voxels_are_anisotropic_by_default(self, tmp_path):
        """z = physical spacing, in-plane = grid. The whole point of the module."""
        csv, paths = _manifest(tmp_path, 3)
        sections = {str(p): _section() for p in paths}

        sdata = build_niche_volume(csv, sections, z_spacing_um=12.0)

        from spatialdata.transformations import get_transformation

        for element in (
            sdata.labels["tissue_labels"],
            sdata.images["niche_probabilities"],
        ):
            m = get_transformation(element, "microns_3d").to_affine_matrix(
                input_axes=("z", "y", "x"), output_axes=("z", "y", "x")
            )
            assert m[0, 0] == pytest.approx(12.0), "z must be section spacing"
            assert m[1, 1] == pytest.approx(32.0)
            assert m[2, 2] == pytest.approx(32.0)

    def test_z_extent_tracks_section_count_not_grid(self, tmp_path):
        """A 3-section stack at 12 um spans 24 um between first and last."""
        csv, paths = _manifest(tmp_path, 3)
        sections = {str(p): _section() for p in paths}
        sdata = build_niche_volume(csv, sections, z_spacing_um=12.0)
        assert sdata.labels["tissue_labels"].shape[0] == 3

    def test_target_voxel_um_forces_isotropy(self, tmp_path):
        csv, paths = _manifest(tmp_path, 2)
        sections = {str(p): _section(shape=(8, 10)) for p in paths}

        sdata = build_niche_volume(
            csv, sections, z_spacing_um=12.0, target_voxel_um=16.0
        )

        from spatialdata.transformations import get_transformation

        m = get_transformation(
            sdata.labels["tissue_labels"], "microns_3d"
        ).to_affine_matrix(
            input_axes=("z", "y", "x"), output_axes=("z", "y", "x")
        )
        # z stays physical; in-plane is resampled to the requested size.
        assert m[0, 0] == pytest.approx(12.0)
        assert m[1, 1] == pytest.approx(16.0)
        # 32 um grid -> 16 um voxels doubles the in-plane extent.
        assert sdata.labels["tissue_labels"].shape[1:] == (16, 20)

    def test_coarsening_via_target_voxel_um_is_refused(self, tmp_path):
        csv, paths = _manifest(tmp_path, 2)
        sections = {str(p): _section() for p in paths}
        with pytest.raises(ValueError, match="Rasterise at grid_um"):
            build_niche_volume(
                csv, sections, z_spacing_um=12.0, target_voxel_um=250.0
            )


class TestElements:
    def test_both_elements_are_built(self, tmp_path):
        csv, paths = _manifest(tmp_path, 2)
        sections = {str(p): _section(winner="immune") for p in paths}
        sdata = build_niche_volume(csv, sections, z_spacing_um=12.0)

        probs = sdata.images["niche_probabilities"]
        labels = sdata.labels["tissue_labels"]

        assert probs.dims == ("c", "z", "y", "x")
        assert list(probs.coords["c"].values) == CLASSES
        assert labels.dims == ("z", "y", "x")
        assert np.asarray(labels.data).dtype == np.uint8
        # immune won everywhere on tissue.
        assert (np.asarray(labels.data) == NICHE_LABEL_INDEX["immune"]).all()

    def test_labels_key_is_tissue_labels_for_downstream_reuse(self, tmp_path):
        """quantification._labels_array and view_volume both hardcode this name."""
        csv, paths = _manifest(tmp_path, 2)
        sections = {str(p): _section() for p in paths}
        sdata = build_niche_volume(csv, sections, z_spacing_um=12.0)
        assert "tissue_labels" in sdata.labels

    def test_class_order_is_recorded_in_attrs(self, tmp_path):
        csv, paths = _manifest(tmp_path, 2)
        sections = {str(p): _section() for p in paths}
        sdata = build_niche_volume(csv, sections, z_spacing_um=12.0)
        assert sdata.attrs["niche_classes"] == CLASSES
        assert sdata.attrs["niche_voxel_um_zyx"] == [12.0, 32.0, 32.0]


class TestValidation:
    def test_shape_mismatch_raises_with_a_pointed_message(self, tmp_path):
        csv, paths = _manifest(tmp_path, 2)
        sections = {
            str(paths[0]): _section(shape=(8, 10)),
            str(paths[1]): _section(shape=(9, 10)),
        }
        with pytest.raises(ValueError, match="registered OME-TIFFs from one VALIS run"):
            build_niche_volume(csv, sections, z_spacing_um=12.0)

    def test_missing_section_raises(self, tmp_path):
        csv, paths = _manifest(tmp_path, 3)
        sections = {str(p): _section() for p in paths[:2]}
        with pytest.raises(KeyError, match="No niche raster provided"):
            build_niche_volume(csv, sections, z_spacing_um=12.0)

    def test_class_mismatch_raises(self, tmp_path):
        csv, paths = _manifest(tmp_path, 2)
        odd = _section()
        odd.classes = ["acellular", "epithelium", "immune", "other"]
        sections = {str(paths[0]): _section(), str(paths[1]): odd}
        with pytest.raises(ValueError, match="has classes"):
            build_niche_volume(csv, sections, z_spacing_um=12.0)


class TestSmoothing:
    def test_smoothing_does_not_invent_tissue(self, tmp_path):
        """Off-tissue voxels stay background after in-plane and z smoothing."""
        csv, paths = _manifest(tmp_path, 3)
        sections = {}
        for p in paths:
            s = _section(shape=(12, 12))
            s.probs[:, :4, :] = np.nan
            s.tile_count[:4, :] = 0
            sections[str(p)] = s

        sdata = build_niche_volume(
            csv, sections, z_spacing_um=12.0, smooth_um=64.0, smooth_z_sigma=1.0
        )
        labels = np.asarray(sdata.labels["tissue_labels"].data)
        assert (labels[:, :4, :] == 0).all()
        assert (labels[:, 4:, :] != 0).all()

    def test_roundtrip_through_zarr(self, tmp_path):
        csv, paths = _manifest(tmp_path, 2)
        sections = {str(p): _section() for p in paths}
        out = tmp_path / "niche_volume.zarr"

        build_niche_volume(csv, sections, z_spacing_um=12.0, output_path=out)

        import spatialdata

        reloaded = spatialdata.read_zarr(str(out))
        assert reloaded.labels["tissue_labels"].shape == (2, 8, 10)
        assert reloaded.images["niche_probabilities"].shape == (4, 2, 8, 10)
