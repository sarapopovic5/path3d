"""The elastic field is estimated over all tissue, not only where every
section's image footprint overlaps."""

import pickle
import subprocess
import sys

import numpy as np
import pytest

import path3d.registration as reg

_TISSUE_RGB = (150, 80, 120)


class _FakeSlide:
    """A VALIS slide whose rigid warp is the identity."""

    def __init__(self, blobs, shape=(100, 100)):
        self.image = np.full((*shape, 3), 255, dtype=np.uint8)
        for r0, r1, c0, c1 in blobs:
            self.image[r0:r1, c0:c1] = _TISSUE_RGB
        self.processed_img_shape_rc = shape

    def warp_img(self, img, non_rigid=True, crop=True, interp_method="bicubic"):
        assert not non_rigid and not crop and interp_method == "nearest"
        return img


def _bbox(mask):
    rows, cols = np.nonzero(mask)
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1


def test_box_covers_tissue_shared_by_two_sections_plus_padding():
    slides = [_FakeSlide([(20, 40, 30, 50)]), _FakeSlide([(20, 40, 30, 50)])]
    mask = reg._tissue_bbox_mask(slides, (100, 100))
    # padding is 2% of the frame + 2 px = 4 px each side
    assert _bbox(mask) == (16, 44, 26, 54)
    assert set(np.unique(mask)) == {0, 255}


def test_tissue_seen_in_one_section_only_is_ignored():
    slides = [
        _FakeSlide([(20, 40, 30, 50)]),
        _FakeSlide([(20, 40, 30, 50), (70, 90, 70, 90)]),
    ]
    assert _bbox(reg._tissue_bbox_mask(slides, (100, 100))) == (16, 44, 26, 54)


def test_box_spans_fragments_that_never_all_overlap():
    # upper fragment in sections 0-1, lower fragment in sections 1-2: no
    # region is shared by all three, which is what the "overlap" box needs
    slides = [
        _FakeSlide([(10, 30, 10, 30)]),
        _FakeSlide([(10, 30, 10, 30), (60, 80, 60, 80)]),
        _FakeSlide([(60, 80, 60, 80)]),
    ]
    assert _bbox(reg._tissue_bbox_mask(slides, (100, 100))) == (6, 84, 6, 84)


def test_no_tissue_falls_back_to_the_full_frame():
    mask = reg._tissue_bbox_mask([_FakeSlide([]), _FakeSlide([])], (100, 100))
    assert (mask == 255).all()


def test_register_rejects_unknown_mask(tmp_path):
    with pytest.raises(ValueError, match="non_rigid_mask"):
        reg.register([], tmp_path, non_rigid_mask="box")


def test_valis_subclass_only_replaces_the_mask_and_pickles():
    valis_registration = pytest.importorskip("valis.registration")
    cls = reg._tissue_mask_valis_cls()
    assert issubclass(cls, valis_registration.Valis)
    assert {k for k in vars(cls) if not k.startswith("__")} == {"create_non_rigid_reg_mask"}
    # VALIS pickles the registrar at the end of register(); it must load
    # again in a process that has not built the class yet
    assert pickle.loads(pickle.dumps(cls)) is cls
    load = "import pickle, sys; print(pickle.loads(sys.stdin.buffer.read()).__name__)"
    done = subprocess.run([sys.executable, "-c", load], input=pickle.dumps(cls),
                          capture_output=True, check=True)
    assert done.stdout.decode().strip().endswith("_TissueMaskValis")
