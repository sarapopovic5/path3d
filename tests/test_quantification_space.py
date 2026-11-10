"""Quantification reports tissue classes only: no output for `space`."""

import numpy as np
import pandas as pd
import pytest

from path3d.quantification import (
    compute_density,
    compute_neighborhood_features,
    compute_volumetrics,
    run_quantification,
)
from path3d.volume import build_volume

VOXEL_UM = 8.0


@pytest.fixture
def sdata(make_manifest):
    """Four 8 um sections, 40 x 40 voxels: columns 0-19 space, 20-29
    Epithelium, 30-39 Stroma. Nuclei on a grid over the whole plane, so some
    land in space."""
    from path3d.slide_io import load_manifest

    manifest = make_manifest(4)
    plane = np.zeros((40, 40), dtype=np.uint8)
    plane[:, 20:30] = 1
    plane[:, 30:] = 2
    label_maps = {str(p): plane.copy() for p in load_manifest(manifest)}
    yy, xx = np.meshgrid(np.arange(4, 320, 16.0), np.arange(4, 320, 16.0), indexing="ij")
    feats = {
        key: pd.DataFrame({
            "reg_y_um": yy.ravel(),
            "reg_x_um": xx.ravel(),
            "feret_diameter_um": np.full(yy.size, 8.0),
        })
        for key in label_maps
    }
    return build_volume(
        manifest, label_maps, nuclear_features=feats,
        target_voxel_um=VOXEL_UM, section_mpp=VOXEL_UM,
    )


def test_volumetrics_has_no_space_row_and_fractions_are_of_tissue(sdata):
    df = compute_volumetrics(sdata, voxel_um=VOXEL_UM)
    assert list(df["class_name"]) == ["Epithelium", "Stroma", "rbc", "Intraluminal secretion"]
    assert df["volume_fraction"].sum() == pytest.approx(1.0)
    frac = dict(zip(df["class_name"], df["volume_fraction"]))
    assert frac["Epithelium"] == pytest.approx(0.5)
    assert frac["Stroma"] == pytest.approx(0.5)


def test_density_has_no_space_row(sdata):
    df = compute_density(sdata, voxel_um=VOXEL_UM)
    assert "space" not in set(df["class_name"])
    n_total = sdata.tables["nuclei"].n_obs
    assert 0 < df["n_nuclei"].sum() < n_total  # nuclei in space are not counted


def test_neighbourhood_shares_are_over_tissue_nuclei(sdata):
    df = compute_neighborhood_features(sdata, voxel_um=VOXEL_UM, radii_um=(50.0,))
    assert "composition_frac_0_r50" not in df.columns
    assert "nn_dist_um_class_0" not in df.columns
    shares = df[[f"composition_frac_{c}_r50" for c in (1, 2, 3, 4)]]
    total = shares.sum(axis=1, min_count=1)
    assert np.allclose(total.dropna(), 1.0)
    # Space nuclei still count as neighbours.
    assert (df["n_neighbors_r50"] > 0).all()


def test_report_and_csvs_never_mention_space(sdata, tmp_path):
    report = run_quantification(sdata, tmp_path, voxel_um=VOXEL_UM).read_text()
    assert "failed" not in report
    assert "space" not in report.split("## Run log")[0].replace("non-space", "")
    lesions = pd.read_csv(tmp_path / "lesions.csv")
    shares = lesions[[f"composition_frac_{c}" for c in (1, 2, 3, 4)]]
    assert "composition_frac_0" not in lesions.columns
    assert np.allclose(shares.sum(axis=1), 1.0)
    for name in ("volumetrics.csv", "density.csv"):
        assert "space" not in (tmp_path / name).read_text()
    for path in tmp_path.glob("radial_profile_lesion*.csv"):
        assert "composition_frac_0" not in pd.read_csv(path).columns
