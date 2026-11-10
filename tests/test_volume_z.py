"""Sections sit at their physical z; gaps stay gaps."""

import warnings

import numpy as np
import pandas as pd
import pytest

from path3d.slide_io import create_manifest, load_manifest_section_numbers
from path3d.volume import build_volume


# The pilot: 4 um sections, every second one stained, so 8 um apart.
PILOT = {"section_thickness_um": 4.0, "sampling_interval": 2}


def _build(manifest, label_maps, **kwargs):
    kwargs.setdefault("section_mpp", kwargs.get("target_voxel_um", 8.0))
    return build_volume(manifest, label_maps, **kwargs)


def _first_voxels(sdata):
    vol = np.asarray(sdata.labels["tissue_labels"].data)
    return [int(v) for v in vol[:, 0, 0]]


def _voxel_scale(sdata):
    from spatialdata.transformations import get_transformation

    affine = get_transformation(sdata.labels["tissue_labels"], "microns_3d")
    matrix = affine.to_affine_matrix(input_axes=("z", "y", "x"),
                                     output_axes=("z", "y", "x"))
    return np.diag(matrix)[:3].tolist()


def test_voxel_is_thickness_times_interval_on_every_axis(make_manifest, make_label_maps):
    manifest = make_manifest(4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sdata = _build(manifest, make_label_maps(manifest, shape=(40, 40)),
                       section_mpp=2.0, **PILOT)
    # no "not given" or empty-plane warning (dependency notices are fine)
    assert not [w for w in caught if "not given" in str(w.message)
                or "z-planes" in str(w.message)]
    assert _voxel_scale(sdata) == [8.0, 8.0, 8.0]
    # 40 px at 2 um/px is 80 um, ten 8 um voxels in-plane
    assert sdata.labels["tissue_labels"].shape == (4, 10, 10)
    assert _first_voxels(sdata) == [0, 1, 2, 3]
    assert sdata.attrs["z_planes"]["section_index"] == [0, 1, 2, 3]
    assert sdata.attrs["voxel"] == {
        "voxel_um": 8.0, "section_thickness_um": 4.0, "sampling_interval": 2,
    }


def test_every_section_stained_gives_thickness_sized_voxels(make_manifest, make_label_maps):
    manifest = make_manifest(3)
    sdata = _build(manifest, make_label_maps(manifest), section_mpp=5.0,
                   section_thickness_um=5.0, sampling_interval=1)
    assert _voxel_scale(sdata) == [5.0, 5.0, 5.0]
    assert sdata.attrs["z_planes"]["z_um"] == [0.0, 5.0, 10.0]


def test_voxel_that_is_not_the_section_spacing_raises(make_manifest, make_label_maps):
    manifest = make_manifest(2)
    with pytest.raises(ValueError, match="voxel is that spacing"):
        _build(manifest, make_label_maps(manifest), target_voxel_um=12.0, **PILOT)
    # the same value is accepted
    _build(manifest, make_label_maps(manifest), target_voxel_um=8.0, **PILOT)


@pytest.mark.parametrize("given", [
    {"section_thickness_um": 4.0},
    {"sampling_interval": 2},
])
def test_thickness_and_interval_go_together(make_manifest, make_label_maps, given):
    manifest = make_manifest(2)
    with pytest.raises(ValueError, match="together"):
        _build(manifest, make_label_maps(manifest), **given)


@pytest.mark.parametrize("given, match", [
    ({"section_thickness_um": 0.0, "sampling_interval": 2}, "must be > 0"),
    ({"section_thickness_um": 4.0, "sampling_interval": 0}, "whole number >= 1"),
    ({"section_thickness_um": 4.0, "sampling_interval": 1.5}, "whole number >= 1"),
])
def test_bad_thickness_or_interval_raises(make_manifest, make_label_maps, given, match):
    manifest = make_manifest(2)
    with pytest.raises(ValueError, match=match):
        _build(manifest, make_label_maps(manifest), **given)


def test_manifest_thickness_that_is_the_cut_thickness_raises(make_manifest, make_label_maps):
    # thickness_um holds the distance between rows (8), not the cut thickness (4)
    manifest = make_manifest(3, thickness_um=[4, 4, 4])
    with pytest.raises(ValueError, match="disagrees with the section spacing"):
        _build(manifest, make_label_maps(manifest), **PILOT)
    manifest = make_manifest(3, thickness_um=[8, 8, 8])
    sdata = _build(manifest, make_label_maps(manifest), **PILOT)
    assert sdata.attrs["z_planes"]["z_um"] == [0.0, 8.0, 16.0]


def test_default_voxel_is_12um(make_manifest, make_label_maps):
    manifest = make_manifest(2)
    with pytest.warns(UserWarning, match="12.0 um apart"):
        sdata = _build(manifest, make_label_maps(manifest), section_mpp=12.0)
    assert _voxel_scale(sdata) == [12.0, 12.0, 12.0]
    assert sdata.attrs["voxel"]["section_thickness_um"] is None


def test_8um_sections_on_a_12um_grid_are_resampled_not_stretched(
    make_manifest, make_label_maps
):
    # 7 sections 8 um apart span 48 um; a 12 um grid over 0..48 has 5 planes.
    # Before, each section became one plane, so the stack spanned 72 um.
    manifest = make_manifest(7, thickness_um=[8] * 7)
    sdata = _build(manifest, make_label_maps(manifest), target_voxel_um=12.0)
    # plane z 0, 12, 24, 36, 48 -> nearest section z 0, 8|16, 24, 32|40, 48
    sources = sdata.attrs["z_planes"]["section_index"]
    assert len(sources) == 5
    assert sources[0] == 0 and sources[2] == 3 and sources[4] == 6
    assert sources[1] in (1, 2) and sources[3] in (4, 5)


def test_missing_section_numbers_leave_empty_planes(make_manifest, make_label_maps):
    # sections 12 and 13 were removed, as two sections were in the pilot
    manifest = make_manifest(4, section_number=[10, 11, 14, 15])
    with pytest.warns(UserWarning, match="2 of 6 z-planes"):
        sdata = _build(manifest, make_label_maps(manifest), **PILOT)
    assert sdata.attrs["z_planes"]["section_index"] == [0, 1, -1, -1, 2, 3]
    assert sdata.attrs["z_planes"]["z_um"] == [0.0, 8.0, 16.0, 24.0, 32.0, 40.0]
    vol = np.asarray(sdata.labels["tissue_labels"].data)
    assert (vol[2:4] == 0).all()
    assert _first_voxels(sdata)[4:] == [2, 3]


def test_nuclei_take_the_same_physical_z_as_their_labels(make_manifest, make_label_maps):
    manifest = make_manifest(3, section_number=[1, 2, 5])
    label_maps = make_label_maps(manifest)
    feats = {
        key: pd.DataFrame({"reg_y_um": [10.0], "reg_x_um": [20.0]})
        for key in label_maps
    }
    with pytest.warns(UserWarning, match="z-planes"):
        sdata = _build(manifest, label_maps, nuclear_features=feats, **PILOT)
    z_nuclei = sdata.tables["nuclei"].obsm["spatial_3d"][:, 0].tolist()
    assert z_nuclei == [0.0, 8.0, 32.0]
    planes = sdata.attrs["z_planes"]
    assert [planes["z_um"][planes["section_index"].index(k)] for k in range(3)] == z_nuclei


def test_thickness_column_now_moves_labels_too(make_manifest, make_label_maps):
    # thickness_um used to place nuclei only; labels ignored it
    manifest = make_manifest(4, thickness_um=[8, 16, 8, 8])
    with pytest.warns(UserWarning, match="1 of 5 z-planes"):
        sdata = _build(manifest, make_label_maps(manifest), target_voxel_um=8.0)
    assert sdata.attrs["z_planes"]["section_index"] == [0, 1, -1, 2, 3]


def test_unknown_spacing_warns(make_manifest, make_label_maps):
    manifest = make_manifest(2)
    with pytest.warns(UserWarning, match="sampling_interval not given"):
        _build(manifest, make_label_maps(manifest), target_voxel_um=12.0)


def test_create_manifest_writes_section_numbers(tmp_path):
    slides = tmp_path / "slides"
    slides.mkdir()
    for name in ("block_101.czi", "block_102.czi", "block_105.czi"):
        (slides / name).touch()
    manifest = create_manifest(slides, tmp_path / "m.csv",
                               section_numbers=[101, 102, 105])
    assert load_manifest_section_numbers(manifest) == [101, 102, 105]
    assert load_manifest_section_numbers(create_manifest(slides, tmp_path / "plain.csv")) is None


def test_section_numbers_must_increase(make_manifest):
    with pytest.raises(ValueError, match="increase strictly"):
        load_manifest_section_numbers(make_manifest(3, section_number=[1, 3, 3]))


def test_plane_record_is_written_with_the_volume(tmp_path, make_manifest, make_label_maps):
    from spatialdata import read_zarr

    manifest = make_manifest(3, section_number=[1, 2, 4])
    with pytest.warns(UserWarning, match="z-planes"):
        _build(manifest, make_label_maps(manifest),
               output_path=tmp_path / "vol.zarr", **PILOT)
    attrs = read_zarr(tmp_path / "vol.zarr").attrs
    assert attrs["z_planes"]["section_index"] == [0, 1, -1, 2]
    assert attrs["voxel"]["voxel_um"] == 8.0
