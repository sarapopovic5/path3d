"""segment_first: segment the original slide at the training resolution, then
warp the label map onto the registered canvas."""

import numpy as np
import pytest

import path3d.pipeline as pl
import path3d.registration as reg


class _Pyramid:
    def __init__(self, mpp0, downsamples=(1.0, 2.0, 4.0, 8.0)):
        self._mpp0 = mpp0
        self.level_downsamples = downsamples

    def get_mpp(self):
        return self._mpp0


@pytest.mark.parametrize("mpp0, expected", [
    (0.2201737, 0.4403474),  # pilot CZIs: level 1, where the training patches came from
    (0.25, 0.25),            # level 1 (0.5) is >1% coarser than 0.4402, so stay at level 0
    (0.11, 0.44),            # a 4x-finer scanner reaches the training scale at level 2
    (0.5, 0.5),              # nothing fine enough: level 0, never upsampled
])
def test_training_level_is_the_coarsest_not_coarser_than_training(mpp0, expected):
    assert pl._training_level_mpp(_Pyramid(mpp0)) == pytest.approx(expected)


class _IdentitySlide:
    """A VALIS slide whose registration is the identity, on a canvas twice the
    size of the label map (as 0.5 um canvas from a 1.0 um map)."""

    def __init__(self, shape_rc):
        self.M = np.eye(3)
        self.bk_dxdy = None
        self.processed_img_shape_rc = shape_rc
        self.reg_img_shape_rc = shape_rc
        self.aligned_slide_shape_rc = shape_rc

    def get_crop_method(self, crop):
        return False


def test_label_warp_keeps_class_values_and_lands_on_the_canvas():
    pytest.importorskip("valis")
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, size=(40, 50)).astype(np.uint8)
    labels[:10, :10] = 4  # a block big enough to survive any resampling intact

    slide = _IdentitySlide((40, 50))
    warped = reg.warp_label_map_to_registered(
        slide, np.kron(labels, np.ones((2, 2), np.uint8)),  # the map at 2x the processed size
        reg_mpp=1.0, target_mpp=0.5, non_rigid=False, crop=False,
    )

    canvas = tuple(reg._registered_canvas_shape_rc(slide, 1.0, 0.5))
    assert warped.shape == canvas == (80, 100)
    assert warped.dtype == np.uint8
    assert set(np.unique(warped)) <= set(range(5))  # nearest: no in-between classes
    assert (warped[2:18, 2:18] == 4).all()


@pytest.mark.parametrize("bad", [np.zeros((4, 4, 3)), np.full((4, 4), 300)])
def test_label_warp_rejects_non_label_input(bad):
    with pytest.raises(ValueError):
        reg.warp_label_map_to_registered(_IdentitySlide((4, 4)), bad)
