"""Tests for the whole-stack runner, from per-section tile CSVs onward.

Prediction itself needs the gated encoder, so these start from the files
``predict_sections`` writes -- ``niches/tiles/NNNN_tiles_niches.csv`` and
``NNNN_meta.json`` -- and check the run directory built from them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from path3d.niches.pipeline import predict_sections
from path3d.niches.predict import prob_columns
from path3d.niches.run import build_niche_run

pytest.importorskip("spatialdata")

CLASSES = ["acellular", "epithelium", "immune", "stroma"]
MPP, STEP = 0.5, 64  # 32 um tiles at 0.5 um/px
NY, NX = 10, 12


def _fake_run(tmp_path, n_sections, *, stride=1, skip=()):
    """Manifest + per-section tile CSVs as predict_sections would leave them."""
    tiles_dir = tmp_path / "out" / "niches" / "tiles"
    tiles_dir.mkdir(parents=True)
    rows = ["section_index,filename,path"]
    rng = np.random.default_rng(0)
    gy, gx = np.mgrid[0:NY, 0:NX]
    on = (gx >= 2) & (gx < 10)
    for i in range(n_sections):
        rows.append(f"{i},s{i}.czi,/cluster/s{i}.czi")
        if i in skip:
            continue
        cx = ((gx + 0.5) * STEP)[on].astype(int)
        cy = ((gy + 0.5) * STEP)[on].astype(int)
        df = pd.DataFrame({"cx_px": cx, "cy_px": cy, "x0_px": cx - 112, "y0_px": cy - 112})
        for col, p in zip(prob_columns(CLASSES), rng.dirichlet([1, 2, 0.3, 1], len(cx)).T):
            df[col] = p
        df.to_csv(tiles_dir / f"{i:04d}_tiles_niches.csv", index=False)
        meta = {
            "mpp": MPP,
            "canvas_wh": [NX * STEP, NY * STEP],
            "classes": CLASSES,
            "n_tiles": len(df),
            "tissue_frac": float(on.mean()),
            "stride": stride,
        }
        (tiles_dir / f"{i:04d}_meta.json").write_text(json.dumps(meta))
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(rows) + "\n")
    return manifest, tmp_path / "out"


def _nuclei(n_sections, per_section=5):
    k = np.repeat(np.arange(n_sections), per_section)
    return pd.DataFrame(
        {
            "z_um": k * 8.0,
            "y_um": 100.0,
            "x_um": 200.0,
            "instance_id": np.arange(1, len(k) + 1),
            "section_index": k,
            "region": "tissue_labels",
            "area_um2": 30.0,
        }
    )


def _build(manifest, out, **kw):
    return build_niche_run(
        manifest, out / "no_registered", out,
        z_spacing_um=8.0, voxel_um=32.0, predict=False, **kw,
    )


class TestRunDirectory:
    def test_writes_the_full_volume_dpt_layout(self, tmp_path):
        manifest, out = _fake_run(tmp_path, 8)
        zarr_path = _build(manifest, out)

        assert zarr_path == out / "volume_niches_32um.zarr"
        assert zarr_path.is_dir()
        assert (out / "manifest_thickness_8um.csv").exists()
        assert (out / "volume_build_metadata.json").exists()
        assert (out / "quantification" / "soft_volumetrics.csv").exists()
        assert (out / "quantification" / "per_section_profile.csv").exists()
        for i in range(8):
            assert (out / "label_maps" / f"{i:04d}_labels.png").exists()
            assert (out / "label_maps" / f"{i:04d}_labels_rgb.png").exists()

        thickness = pd.read_csv(out / "manifest_thickness_8um.csv")["thickness_um"]
        assert (thickness == 8.0).all()

    def test_volume_is_32um_cubes_of_4_sections(self, tmp_path):
        manifest, out = _fake_run(tmp_path, 9)
        with pytest.warns(UserWarning, match="left out"):
            zarr_path = _build(manifest, out)

        import spatialdata

        sdata = spatialdata.read_zarr(str(zarr_path))
        assert sdata.labels["tissue_labels"].shape == (2, NY, NX)
        assert sdata.attrs["niche_voxel_um_zyx"] == [32.0, 32.0, 32.0]

        meta = json.loads((out / "volume_build_metadata.json").read_text())
        assert meta["sections_per_voxel"] == 4
        assert meta["dropped_sections"] == [8]
        assert meta["volume_shape_zyx"] == [2, NY, NX]
        assert meta["tissue_type"] == "HGSC_niches"
        assert [s["in_volume"] for s in meta["sections"]] == [True] * 8 + [False]

    def test_nuclei_are_attached_minus_dropped_sections(self, tmp_path):
        manifest, out = _fake_run(tmp_path, 9)
        parquet = tmp_path / "nuclei.parquet"
        _nuclei(9).to_parquet(parquet)

        with pytest.warns(UserWarning, match="left out"):
            zarr_path = _build(manifest, out, nuclei_parquet=parquet)

        import spatialdata

        nuclei = spatialdata.read_zarr(str(zarr_path)).tables["nuclei"]
        assert nuclei.n_obs == 40
        assert 8 not in set(nuclei.obs["section_index"])

    def test_rebuild_overwrites_the_previous_volume(self, tmp_path):
        manifest, out = _fake_run(tmp_path, 4)
        _build(manifest, out)
        _build(manifest, out, smooth_um=64.0)
        meta = json.loads((out / "volume_build_metadata.json").read_text())
        assert meta["smooth_um"] == 64.0


class TestMissingSections:
    def test_missing_section_stops_the_build(self, tmp_path):
        manifest, out = _fake_run(tmp_path, 4, skip={2})
        with pytest.raises(RuntimeError, match=r"\[2\].*resubmit"):
            _build(manifest, out)

    def test_allow_missing_leaves_it_empty(self, tmp_path):
        manifest, out = _fake_run(tmp_path, 4, skip={2})
        _build(manifest, out, allow_missing=True)
        meta = json.loads((out / "volume_build_metadata.json").read_text())
        assert meta["missing_sections"] == [2]


class TestStrideGuard:
    def test_preview_tiles_are_not_reused_by_a_full_run(self, tmp_path):
        """A stride-4 preview on disk must not pass for a finished section."""
        manifest, out = _fake_run(tmp_path, 2, stride=4)
        with pytest.raises(ValueError, match="different output directories"):
            predict_sections(manifest, tmp_path / "registered", out / "niches", stride=1)
