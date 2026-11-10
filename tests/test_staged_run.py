"""The staged run (register -> sections array -> volume) does the same work as
run_pipeline in one process, split so the GPU stage can run as a Slurm array."""

import numpy as np
import pandas as pd
import pytest

import path3d.pipeline as pl
import path3d.registration as reg

_LEVEL0_RC = (600, 620)


class _Slide:
    """An open slide: level-0 size and resolution only."""

    level_dimensions = ((_LEVEL0_RC[1], _LEVEL0_RC[0]),)  # (w, h)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_mpp(self):
        return 0.2201


class _ValisSlide:
    """A VALIS slide that shifts points by a per-section offset."""

    def __init__(self, idx):
        self.idx = idx
        self.aligned_slide_shape_rc = (60, 62)

    def warp_xy(self, xy, M=None, slide_level=0, pt_level=0, non_rigid=True, crop=True):
        return np.asarray(xy) + self.idx


@pytest.fixture
def fake_stack(monkeypatch):
    """Replace slide reading, VALIS, segmentation and Cellpose with stand-ins
    whose outputs depend on the section, and count the per-section calls."""
    calls = {"segment": [], "nuclei": []}

    monkeypatch.setattr(pl.io, "open_slide", lambda path: _Slide())
    monkeypatch.setattr(pl.pr, "downsample_to_mpp",
                        lambda slide, mpp: np.full((8, 8, 3), 255, np.uint8))
    monkeypatch.setattr(pl.reg, "register",
                        lambda rows, out, **kw: {r["section_index"]: _ValisSlide(r["section_index"])
                                                 for r in rows})
    monkeypatch.setattr(pl.reg, "build_section_transforms", lambda slides, *a: {})
    monkeypatch.setattr(pl.reg, "load_registered_slides",
                        lambda out, idxs: {i: _ValisSlide(i) for i in idxs})
    monkeypatch.setattr(pl.reg, "warp_and_save_section", lambda slide, src, dst, **kw: dst)

    def predict(path, model, mask, *, section_idx, **kw):
        calls["segment"].append(section_idx)
        return np.full((6, 6), section_idx % 5, np.uint8)

    def detect(path, **kw):
        idx = int(path.stem[1:])  # s{i}.czi
        calls["nuclei"].append(idx)
        return pd.DataFrame({"centroid_y": [100.0 + idx], "centroid_x": [200.0]})

    monkeypatch.setattr(pl.sg, "predict_section", predict)
    monkeypatch.setattr(pl.nd, "detect_nuclei_section", detect)
    return calls


_MODEL = object()  # anything but a path is used as the model as-is


def _outputs(out):
    labels = {p.name: p.read_bytes() for p in sorted((out / "label_maps").glob("*.png"))}
    nuclei = {p.name: pd.read_csv(p) for p in sorted((out / "nuclei").glob("*.csv"))}
    return labels, nuclei


@pytest.mark.parametrize("n_sections, n_tasks", [(7, 3), (5, 5), (3, 8)])
def test_tasks_split_every_section_exactly_once(n_sections, n_tasks):
    shares = [list(pl.task_sections(n_sections, t, n_tasks)) for t in range(n_tasks)]
    assert sorted(i for share in shares for i in share) == list(range(n_sections))


def test_tasks_are_interleaved_through_the_depth():
    assert list(pl.task_sections(10, 1, 3)) == [1, 4, 7]


@pytest.mark.parametrize("task_index, n_tasks", [(3, 3), (-1, 3), (0, 0)])
def test_a_task_index_outside_the_array_raises(task_index, n_tasks):
    with pytest.raises(ValueError, match="task_index"):
        pl.task_sections(10, task_index, n_tasks)


def test_staged_run_writes_what_run_pipeline_writes(tmp_path, make_manifest, fake_stack, monkeypatch):
    manifest = make_manifest(5)
    # The volume itself is covered by test_volume_from_run; here only that both
    # paths hand it the same inputs.
    built = []
    monkeypatch.setattr(pl, "_build_and_quantify", lambda out, m, **kw: built.append(kw))

    single = tmp_path / "single"
    pl.run_pipeline(manifest, single, seg_model=_MODEL, num_classes=5, seg_device="cpu",
                    section_thickness_um=4, sampling_interval=2)

    staged = tmp_path / "staged"
    pl.run_register_stage(manifest, staged, section_thickness_um=4, sampling_interval=2)
    for task in range(2):
        pl.run_sections_stage(staged, task_index=task, n_tasks=2, seg_model=_MODEL,
                              num_classes=5, seg_device="cpu")
    pl.run_volume_stage(staged)

    single_labels, single_nuclei = _outputs(single)
    staged_labels, staged_nuclei = _outputs(staged)
    assert staged_labels == single_labels and len(staged_labels) == 10  # labels + RGB
    assert staged_nuclei.keys() == single_nuclei.keys()
    for name in single_nuclei:
        pd.testing.assert_frame_equal(staged_nuclei[name], single_nuclei[name])
    for kw in built:
        kw.pop("volume_path")
    assert built[0] == built[1]
    assert built[1]["voxel_um"] == 8.0 and built[1]["sampling_interval"] == 2


def test_a_rerun_task_skips_finished_sections(tmp_path, make_manifest, fake_stack):
    manifest = make_manifest(4)
    out = tmp_path / "run"
    pl.run_register_stage(manifest, out)
    pl.run_sections_stage(out, task_index=0, n_tasks=2, seg_model=_MODEL, num_classes=5,
                          seg_device="cpu")
    assert fake_stack["segment"] == [0, 2]

    # Section 2's nuclei table never landed, as if the job died there.
    (out / "nuclei" / "0002_nuclei.csv").unlink()
    results = pl.run_sections_stage(out, task_index=0, n_tasks=2, seg_model=_MODEL,
                                    num_classes=5, seg_device="cpu")
    assert [r["section_index"] for r in results] == [2]
    assert fake_stack["segment"] == [0, 2, 2]


def test_sections_before_register_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="register stage"):
        pl.run_sections_stage(tmp_path, seg_device="cpu")


def test_volume_names_the_sections_still_missing(tmp_path, make_manifest, fake_stack):
    manifest = make_manifest(4)
    out = tmp_path / "run"
    pl.run_register_stage(manifest, out)
    pl.run_sections_stage(out, task_index=0, n_tasks=2, seg_model=_MODEL, num_classes=5,
                          seg_device="cpu")
    with pytest.raises(FileNotFoundError, match=r"sections \[1, 3\]"):
        pl.run_volume_stage(out)


def test_register_refuses_an_existing_volume(tmp_path, make_manifest):
    out = tmp_path / "run"
    (out / "volume_8um.zarr").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        pl.run_register_stage(make_manifest(2), out, section_thickness_um=4, sampling_interval=2)


class _Registrar:
    slide_dict = {"0000": "slide 0", "0001.png": "slide 1", "0002": "slide 2"}


def test_registered_slides_reload_by_section(tmp_path, monkeypatch):
    import valis.registration

    data = tmp_path / "images" / "data"
    data.mkdir(parents=True)
    (data / "images_registrar.pickle").write_bytes(b"")
    loaded = []
    monkeypatch.setattr(valis.registration, "load_registrar",
                        lambda f: loaded.append(f) or _Registrar())

    assert reg.load_registered_slides(tmp_path, [1, 2]) == {1: "slide 1", 2: "slide 2"}
    assert loaded == [str(data / "images_registrar.pickle")]


def test_reloading_without_a_registration_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="found 0"):
        reg.load_registered_slides(tmp_path, [0])
