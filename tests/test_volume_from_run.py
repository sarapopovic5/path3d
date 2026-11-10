"""build_volume_from_run rebuilds the volume from a run directory on disk."""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from path3d.slide_io import load_manifest
from path3d.volume import build_volume_from_run


# The pilot: 4 um sections, every second one stained, so 8 um apart.
PILOT = {"section_thickness_um": 4.0, "sampling_interval": 2}


def _write_run(run_dir, manifest, *, metadata=None, shape=(40, 40), seg_mpp=2.0):
    """A minimal run_pipeline output directory: section k's label PNG is
    filled with class k % 5, and its nuclei CSV holds two nuclei already in
    the registered canvas."""
    (run_dir / "label_maps").mkdir(parents=True)
    (run_dir / "nuclei").mkdir()
    for k, _ in enumerate(load_manifest(manifest)):
        Image.fromarray(np.full(shape, k % 5, dtype=np.uint8)).save(
            run_dir / "label_maps" / f"{k:04d}_labels.png"
        )
        pd.DataFrame({
            "centroid_y": [10.0, 20.0], "centroid_x": [10.0, 20.0],
            "reg_y_um": [16.0, 40.0], "reg_x_um": [24.0, 48.0],
            "feret_diameter_um": [6.0, 7.0],
        }).to_csv(run_dir / "nuclei" / f"{k:04d}_nuclei.csv", index=False)
    with open(run_dir / "section_transforms.pkl", "wb") as f:
        pickle.dump({}, f)
    meta = {"manifest_csv": str(manifest), "seg_mpp": seg_mpp,
            "registration_mpp": 8.0, "level0_mpp": 0.22}
    meta.update(metadata or {})
    with open(run_dir / "volume_build_metadata.json", "w") as f:
        json.dump(meta, f)
    return run_dir


def _nuclei_z(sdata):
    return sorted(set(sdata.tables["nuclei"].obsm["spatial_3d"][:, 0].tolist()))


def test_thickness_and_interval_recorded_by_the_run_are_reused(tmp_path, make_manifest):
    manifest = make_manifest(3)
    run_dir = _write_run(tmp_path / "run", manifest, metadata=PILOT)
    sdata = build_volume_from_run(run_dir)
    assert sdata.attrs["voxel"] == {
        "voxel_um": 8.0, "section_thickness_um": 4.0, "sampling_interval": 2,
    }
    # 40 px at 2 um/px is 80 um, ten 8 um voxels in-plane
    assert sdata.labels["tissue_labels"].shape == (3, 10, 10)
    assert _nuclei_z(sdata) == [0.0, 8.0, 16.0]


def test_registered_nucleus_columns_are_used_unsmoothed(tmp_path, make_manifest):
    manifest = make_manifest(3)
    run_dir = _write_run(tmp_path / "run", manifest)
    sdata = build_volume_from_run(run_dir, **PILOT)
    yx = sdata.tables["nuclei"].obsm["spatial_3d"][:, 1:]
    # the empty transforms pickle would raise if the rigid fallback ran
    assert yx.tolist() == [[16.0, 24.0], [40.0, 48.0]] * 3


def test_arguments_override_the_recorded_values(tmp_path, make_manifest):
    manifest = make_manifest(2)
    run_dir = _write_run(tmp_path / "run", manifest, metadata=PILOT)
    sdata = build_volume_from_run(run_dir, section_thickness_um=5.0, sampling_interval=1)
    assert sdata.attrs["voxel"]["voxel_um"] == 5.0
    assert _nuclei_z(sdata) == [0.0, 5.0]


def test_older_runs_without_an_interval_fall_back_to_the_voxel_size(tmp_path, make_manifest):
    # Runs before sampling_interval was recorded stored the z spacing under
    # section_thickness_um; it must not be read as the cut thickness.
    manifest = make_manifest(2)
    run_dir = _write_run(tmp_path / "run", manifest,
                         metadata={"section_thickness_um": 12.0})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sdata = build_volume_from_run(run_dir, target_voxel_um=8.0)
    assert sdata.attrs["voxel"] == {
        "voxel_um": 8.0, "section_thickness_um": None, "sampling_interval": None,
    }


def test_thickness_without_interval_raises(tmp_path, make_manifest):
    manifest = make_manifest(2)
    run_dir = _write_run(tmp_path / "run", manifest)
    with pytest.raises(ValueError, match="together"):
        build_volume_from_run(run_dir, section_thickness_um=4.0)


@pytest.mark.parametrize("given, match", [
    ({"section_thickness_um": 4.0}, "together"),
    ({**PILOT, "voxel_um": 12.0}, "voxel is that spacing"),
])
def test_run_pipeline_rejects_a_bad_voxel_before_any_work(tmp_path, make_manifest, given, match):
    from path3d.pipeline import run_pipeline

    # The manifest's slides do not exist, so reaching them would fail differently.
    manifest = make_manifest(2)
    with pytest.raises(ValueError, match=match):
        run_pipeline(manifest, tmp_path / "out", seg_device="cpu", **given)


def test_a_missing_label_map_raises(tmp_path, make_manifest):
    manifest = make_manifest(3)
    run_dir = _write_run(tmp_path / "run", manifest, metadata=PILOT)
    (run_dir / "label_maps" / "0001_labels.png").unlink()
    with pytest.raises(ValueError, match=r"section indices \[1\]"):
        build_volume_from_run(run_dir)
