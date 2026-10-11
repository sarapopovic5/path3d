"""Segmentation tiles reach the encoder at the checkpoint's training resolution."""

import warnings

import numpy as np
import pytest
import torch

import path3d.segmentation as sg


def test_matched_tile_shows_the_encoder_training_mpp():
    tile = sg.resolution_matched_tile_size(0.5, 0.4402)
    assert tile == 394
    # what the encoder sees after _preprocess resizes the tile to 448
    assert 0.5 * tile / sg._SEG_IMG_SIZE == pytest.approx(0.4402, abs=0.002)
    assert sg.resolution_matched_tile_size(0.4402, 0.4402) == sg._SEG_IMG_SIZE


def test_old_1024_default_was_2p6x_coarser_than_training():
    presented = 0.5 * 1024 / sg._SEG_IMG_SIZE
    assert presented / 0.4402 == pytest.approx(2.597, abs=0.01)


class _FakeSlide:
    """500 x 500 px of uniform pink tissue at 0.5 um/px, one level."""

    level_downsamples = (1.0,)
    level_dimensions = ((500, 500),)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def best_level_for_mpp(self, mpp):
        return 0

    def read_region(self, location, level, size):
        w, h = size
        return np.full((h, w, 3), (200, 120, 160), dtype=np.uint8)


class _ZeroModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return torch.zeros(x.shape[0], self.num_classes, *x.shape[2:])


def _run(monkeypatch, **kwargs):
    """predict_section on the fake slide; returns the raw tiles it batched."""
    seen = []

    def record(tile):
        seen.append(tile.copy())
        return torch.zeros(3, sg._SEG_IMG_SIZE, sg._SEG_IMG_SIZE)

    monkeypatch.setattr(sg, "open_slide", lambda path: _FakeSlide())
    monkeypatch.setattr(sg, "_preprocess", record)
    tissue_everywhere = np.zeros((500, 500), dtype=bool)  # True = background
    sg.predict_section(
        "unused.ome.tiff", _ZeroModel(5), tissue_everywhere, 5,
        section_idx=0, seg_mpp=0.5, device="cpu", **kwargs,
    )
    return seen


def test_predict_section_defaults_to_matched_tiles(monkeypatch):
    # run_pipeline passes no tile_size, so this default is what a run uses
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tiles = _run(monkeypatch)
    assert {t.shape for t in tiles} == {(394, 394, 3)}
    assert not [w for w in caught if "trained at" in str(w.message)]


def test_edge_tiles_are_padded_white_not_black(monkeypatch):
    tiles = _run(monkeypatch)
    # 500 px at stride 394 - 79 = 315: the second tile in each row and
    # column holds 185 px of tissue and 209 px of padding
    edge = tiles[-1]
    assert (edge[:185, :185] == (200, 120, 160)).all()
    assert (edge[185:, :] == 255).all() and (edge[:, 185:] == 255).all()


def test_off_scale_tile_size_warns(monkeypatch):
    with pytest.warns(UserWarning, match="trained at 0.440"):
        _run(monkeypatch, tile_size=1024, overlap=192)


def test_overlap_must_be_smaller_than_tile(monkeypatch):
    with pytest.raises(ValueError, match="overlap"):
        _run(monkeypatch, tile_size=394, overlap=394)
