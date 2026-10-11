"""Nuclei reach the label frame through VALIS's full (rigid + elastic) warp."""

import numpy as np
import pandas as pd
import pytest

import path3d.registration as reg
from path3d.volume import build_volume

# The pilot's aligned canvas at 8 um/px and the 0.5 um/px registered canvas
# the label maps were segmented on (26464 x 27712 px).
_ALIGNED_RC = (1654, 1732)
_CANVAS_RC = (26464, 27712)
_LEVEL0_RC = (60000, 62000)


class _FakeValisSlide:
    """Stands in for a VALIS Slide: scales points from ``pt_level`` to
    ``slide_level`` shape, then shifts them by a fixed (x, y) offset."""

    aligned_slide_shape_rc = _ALIGNED_RC

    def __init__(self, offset_xy=(10.0, 20.0)):
        self.offset_xy = np.asarray(offset_xy)
        self.calls = []

    def warp_xy(self, xy, M=None, slide_level=0, pt_level=0, non_rigid=True, crop=True):
        self.calls.append(
            {"slide_level": slide_level, "pt_level": pt_level,
             "non_rigid": non_rigid, "crop": crop}
        )
        scale_xy = np.asarray(slide_level, float)[::-1] / np.asarray(pt_level, float)[::-1]
        return np.asarray(xy) * scale_xy + self.offset_xy


def _expected_um(yx):
    y, x = yx
    return [
        (y * _CANVAS_RC[0] / _LEVEL0_RC[0] + 20.0) * 0.5,
        (x * _CANVAS_RC[1] / _LEVEL0_RC[1] + 10.0) * 0.5,
    ]


def test_canvas_matches_the_registered_image():
    canvas = reg._registered_canvas_shape_rc(_FakeValisSlide(), reg_mpp=8.0, target_mpp=0.5)
    assert tuple(canvas) == _CANVAS_RC


def test_points_go_through_valis_in_the_image_frame():
    slide = _FakeValisSlide()
    yx = np.array([[0.0, 0.0], [6000.0, 3100.0]])
    out = reg.warp_points_to_registered(
        slide, yx, level0_shape_rc=_LEVEL0_RC, reg_mpp=8.0, target_mpp=0.5
    )
    np.testing.assert_allclose(out, [_expected_um(p) for p in yx])
    # VALIS does the level-0 -> canvas scaling and the crop, exactly as
    # warp_and_save_section does for the image
    assert slide.calls == [
        {"slide_level": _CANVAS_RC, "pt_level": _LEVEL0_RC, "non_rigid": True, "crop": True}
    ]


def test_rigid_only_is_passed_through():
    slide = _FakeValisSlide()
    reg.warp_points_to_registered(
        slide, np.zeros((1, 2)), level0_shape_rc=_LEVEL0_RC, non_rigid=False
    )
    assert slide.calls[0]["non_rigid"] is False


def test_no_points_no_warp():
    slide = _FakeValisSlide()
    out = reg.warp_points_to_registered(slide, np.zeros((0, 2)), level0_shape_rc=_LEVEL0_RC)
    assert out.shape == (0, 2) and slide.calls == []


def test_run_pipeline_writes_registered_columns(tmp_path, make_manifest, monkeypatch):
    import path3d.pipeline as pl

    class _Slide:
        level_dimensions = ((_LEVEL0_RC[1], _LEVEL0_RC[0]),)  # (w, h)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_mpp(self):
            return 0.2201

    manifest = make_manifest(2)
    monkeypatch.setattr(pl.io, "open_slide", lambda path: _Slide())
    monkeypatch.setattr(pl.pr, "downsample_to_mpp",
                        lambda slide, mpp: np.full((8, 8, 3), 255, np.uint8))
    monkeypatch.setattr(pl.reg, "register",
                        lambda rows, out, **kw: {0: _FakeValisSlide(), 1: _FakeValisSlide()})
    monkeypatch.setattr(pl.reg, "warp_and_save_section", lambda slide, src, dst, **kw: dst)
    monkeypatch.setattr(pl.nd, "detect_nuclei_section",
                        lambda path, **kw: pd.DataFrame({"centroid_y": [6000.0],
                                                         "centroid_x": [3100.0]}))

    pl.run_pipeline(manifest, tmp_path / "out", seg_model=None, seg_device="cpu")

    df = pd.read_csv(tmp_path / "out" / "nuclei" / "0001_nuclei.csv")
    np.testing.assert_allclose(df[["reg_y_um", "reg_x_um"]].to_numpy()[0],
                               _expected_um((6000.0, 3100.0)))


def test_build_volume_uses_registered_columns_without_transforms(make_manifest, make_label_maps):
    manifest = make_manifest(2)
    label_maps = make_label_maps(manifest)
    feats = {
        key: pd.DataFrame({
            "centroid_y": [1.0], "centroid_x": [2.0],
            "reg_y_um": [100.0 + k], "reg_x_um": [200.0 + k],
        })
        for k, key in enumerate(label_maps)
    }
    sdata = build_volume(manifest, label_maps, nuclear_features=feats,
                         target_voxel_um=8.0, section_mpp=8.0)
    xyz = sdata.tables["nuclei"].obsm["spatial_3d"]
    np.testing.assert_allclose(xyz, [[0.0, 100.0, 200.0], [8.0, 101.0, 201.0]])


def test_rigid_fallback_is_not_smoothed_across_sections_by_default(
    make_manifest, make_label_maps
):
    from spatialdata.transformations import Affine

    manifest = make_manifest(3)
    label_maps = make_label_maps(manifest)
    shift_y = [0.0, 100.0, 0.0]  # the middle section sits differently on its slide
    transforms = {
        key: Affine(
            np.array([[0, 0, 0], [1, 0, dy], [0, 1, 0], [0, 0, 1]], dtype=float),
            input_axes=("y", "x"), output_axes=("z", "y", "x"),
        )
        for key, dy in zip(label_maps, shift_y)
    }
    feats = {key: pd.DataFrame({"centroid_y": [0.0], "centroid_x": [0.0]}) for key in label_maps}

    def nucleus_y(**kwargs):
        sdata = build_volume(manifest, label_maps, nuclear_features=feats,
                             transforms=transforms, level0_mpp=1.0, registration_mpp=1.0,
                             target_voxel_um=8.0, section_mpp=8.0, **kwargs)
        return sdata.tables["nuclei"].obsm["spatial_3d"][:, 1]

    np.testing.assert_allclose(nucleus_y(), shift_y)
    # the old default averaged neighbours' transforms
    assert nucleus_y(smooth_z_sigma=1.0)[1] < 100.0


def test_rigid_fallback_still_needs_transforms(make_manifest, make_label_maps):
    manifest = make_manifest(1)
    label_maps = make_label_maps(manifest)
    feats = {key: pd.DataFrame({"centroid_y": [0.0], "centroid_x": [0.0]}) for key in label_maps}
    with pytest.raises(ValueError, match="transforms must be provided"):
        build_volume(manifest, label_maps, nuclear_features=feats,
                     target_voxel_um=8.0, section_mpp=8.0)
