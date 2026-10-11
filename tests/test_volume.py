"""volume.build_volume on a synthetic label stack."""

import numpy as np
import pytest

from path3d.volume import build_volume


def _planes(sdata) -> np.ndarray:
    return np.asarray(sdata.labels["tissue_labels"].data)


def test_one_plane_per_section_in_manifest_order(make_manifest, make_label_maps):
    manifest = make_manifest(4)
    sdata = build_volume(
        manifest, make_label_maps(manifest), target_voxel_um=8.0, section_mpp=8.0
    )
    vol = _planes(sdata)
    assert vol.shape == (4, 40, 40)
    assert [int(vol[k, 0, 0]) for k in range(4)] == [0, 1, 2, 3]


def test_labels_are_resampled_in_plane_to_the_voxel_size(make_manifest, make_label_maps):
    manifest = make_manifest(2)
    sdata = build_volume(
        manifest, make_label_maps(manifest), target_voxel_um=8.0, section_mpp=2.0
    )
    assert _planes(sdata).shape == (2, 10, 10)


def test_missing_label_map_raises(make_manifest, make_label_maps):
    manifest = make_manifest(3)
    label_maps = make_label_maps(manifest)
    label_maps.pop(next(iter(label_maps)))
    with pytest.raises(KeyError, match="No label map"):
        build_volume(manifest, label_maps, target_voxel_um=8.0, section_mpp=8.0)
