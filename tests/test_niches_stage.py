"""Niche stages of the staged run: a native-resolution warp per section, niche
predictions per array task, then the niche volume beside the DPT one."""

import json

import pandas as pd
import pytest
from test_staged_run import _MODEL, fake_stack  # noqa: F401  (fixture)

import path3d.niches.pipeline as npl
import path3d.niches.predict as npredict
import path3d.niches.run as nrun
import path3d.pipeline as pl


@pytest.fixture
def niche_calls(fake_stack, monkeypatch):  # noqa: F811
    """Native warps that write a file, niche prediction that writes its tile
    CSV + meta, and a niche volume build that only records its arguments."""
    calls = {"warp_mpp": {}, "predict": [], "build": [], "encoders": 0, "on_disk": []}

    def load_encoder(device=None):
        calls["encoders"] += 1
        return object()

    def warp(slide, src, dst, *, target_mpp, **kw):
        if "registered_native" in str(dst):
            calls["warp_mpp"][slide.idx] = target_mpp
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"tiff")
        return dst

    def predict(manifest, registered_dir, out, *, step_um, section_indices, encoder, **kw):
        assert encoder is not None
        tiles = out / "tiles"
        tiles.mkdir(parents=True, exist_ok=True)
        # how many native warps exist while this section is predicted
        calls["on_disk"].append(len(list(registered_dir.glob("*.ome.tiff"))))
        for idx in section_indices:
            assert (registered_dir / f"{idx:04d}.ome.tiff").exists()
            calls["predict"].append(idx)
            (tiles / f"{idx:04d}_tiles_niches.csv").write_text("x\n")
            (tiles / f"{idx:04d}_meta.json").write_text(json.dumps({"step_um": step_um}))

    def build(manifest, registered_dir, out, **kw):
        calls["build"].append(kw)
        return out / "volume.zarr"

    monkeypatch.setattr(pl.reg, "warp_and_save_section", warp)
    monkeypatch.setattr(npl, "predict_sections", predict)
    monkeypatch.setattr(npredict, "load_encoder", load_encoder)
    monkeypatch.setattr(nrun, "build_niche_run", build)
    return calls


def _registered(make_manifest, tmp_path, n=4):
    out = tmp_path / "run"
    pl.run_register_stage(make_manifest(n), out, section_thickness_um=4, sampling_interval=2)
    return out


def test_niche_mpp_is_native_but_never_finer_than_training():
    assert pl.niche_mpp(0.22017) == 0.22017
    assert pl.niche_mpp(0.12) == pytest.approx(0.2201)
    assert pl.niche_mpp(None) == pytest.approx(0.2201)


def test_niches_stage_warps_at_native_resolution_and_predicts_its_share(
    tmp_path, make_manifest, niche_calls,
):
    out = _registered(make_manifest, tmp_path)
    done = pl.run_niches_stage(out, task_index=1, n_tasks=2, step_um=32)
    assert done == [1, 3]
    assert niche_calls["predict"] == [1, 3]
    # _Slide.get_mpp is 0.2201: the niche copy is NOT warped at SEG_MPP
    assert niche_calls["warp_mpp"] == {1: 0.2201, 3: 0.2201}
    # one section at a time: a single native warp on disk at any moment, each
    # dropped once its tiles landed, no partial left behind, one encoder load
    assert niche_calls["on_disk"] == [1, 1]
    assert list((out / "registered_native").iterdir()) == []
    assert niche_calls["encoders"] == 1


def test_a_rerun_niches_task_skips_predicted_sections(tmp_path, make_manifest, niche_calls):
    out = _registered(make_manifest, tmp_path)
    pl.run_niches_stage(out, task_index=0, n_tasks=2, step_um=32)
    (out / "neighbourhoods" / "niches" / "tiles" / "0002_meta.json").unlink()
    assert pl.run_niches_stage(out, task_index=0, n_tasks=2, step_um=32) == [2]
    assert niche_calls["predict"] == [0, 2, 2]


def test_keep_warp_keeps_the_native_copy(tmp_path, make_manifest, niche_calls):
    out = _registered(make_manifest, tmp_path, n=2)
    pl.run_niches_stage(out, step_um=32, keep_warp=True)
    names = sorted(p.name for p in (out / "registered_native").iterdir())
    assert names == ["0000.ome.tiff", "0001.ome.tiff"]


def test_a_bad_niche_cube_fails_before_any_gpu_work(tmp_path, make_manifest, niche_calls):
    out = _registered(make_manifest, tmp_path)  # sections 8 um apart
    with pytest.raises(ValueError, match="whole multiple"):
        pl.run_niches_stage(out, step_um=32, niche_voxel_um=12)
    with pytest.raises(ValueError, match="coarser than the tile step"):
        pl.run_niches_stage(out, step_um=8, niche_voxel_um=16)
    assert niche_calls["warp_mpp"] == {} and niche_calls["predict"] == []


def test_niches_volume_uses_the_run_spacing_the_predicted_step_and_its_nuclei(
    tmp_path, make_manifest, niche_calls,
):
    out = _registered(make_manifest, tmp_path)
    pl.run_niches_stage(out, step_um=32)
    pd.DataFrame({"x": [1]}).to_parquet(out / "nuclei_8um.parquet")

    pl.run_niches_volume_stage(out, smooth_um=100)
    kw = niche_calls["build"][0]
    assert kw["z_spacing_um"] == 8.0 and kw["step_um"] == 32.0 and kw["voxel_um"] == 32.0
    assert kw["smooth_um"] == 100 and kw["predict"] is False
    assert kw["nuclei_parquet"] == out / "nuclei_8um.parquet"


def test_niches_volume_without_the_dpt_nuclei_builds_without_them(
    tmp_path, make_manifest, niche_calls,
):
    out = _registered(make_manifest, tmp_path)
    pl.run_niches_stage(out, step_um=32)
    pl.run_niches_volume_stage(out)
    assert niche_calls["build"][0]["nuclei_parquet"] is None


def test_niches_volume_before_any_prediction_raises(tmp_path, make_manifest, niche_calls):
    out = _registered(make_manifest, tmp_path)
    with pytest.raises(FileNotFoundError, match="niches stage"):
        pl.run_niches_volume_stage(out)


def test_run_pipeline_with_niches_matches_the_staged_niche_run(
    tmp_path, make_manifest, niche_calls, monkeypatch,
):
    monkeypatch.setattr(pl, "_build_and_quantify", lambda *a, **kw: None)
    manifest = make_manifest(3)
    single = tmp_path / "single"
    pl.run_pipeline(manifest, single, seg_model=_MODEL, num_classes=5, seg_device="cpu",
                    section_thickness_um=4, sampling_interval=2,
                    niches=True, niche_step_um=32)
    assert niche_calls["predict"] == [0, 1, 2]
    assert niche_calls["warp_mpp"] == {0: 0.2201, 1: 0.2201, 2: 0.2201}
    assert niche_calls["on_disk"] == [1, 1, 1] and niche_calls["encoders"] == 1
    assert list((single / "registered_native").iterdir()) == []
    kw = niche_calls["build"][0]
    assert kw["z_spacing_um"] == 8.0 and kw["step_um"] == 32.0
