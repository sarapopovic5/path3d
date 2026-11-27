"""Unit tests for the tile-table -> raster step.

The load-bearing property is that ``rasterize_tiles`` inverts
``predict.tile_grid`` exactly: a tile that ``tile_grid`` placed at grid cell
``(i, j)`` must come back out of the raster at ``(i, j)``. If that drifts, the
3D volume is silently sheared relative to the H&E.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from path3d.config import NICHE_BACKGROUND_INDEX, NICHE_LABEL_INDEX
from path3d.niches.predict import grid_shape, prob_columns, tile_grid
from path3d.niches.rasterize import (
    argmax_labels,
    rasterize_tiles,
    smooth_probs,
)

CLASSES = ["acellular", "epithelium", "immune", "stroma"]


def _tiles_from_grid(canvas, mpp, tissue, probs_per_tile=None):
    """tile_grid output with synthetic probability columns attached."""
    tiles = tile_grid(canvas, mpp, tissue, tile_um=32.0, fov_um=112.0)
    rng = np.random.default_rng(0)
    p = (
        rng.random((len(tiles), len(CLASSES)))
        if probs_per_tile is None
        else np.asarray(probs_per_tile, float)
    )
    p = p / p.sum(1, keepdims=True)
    for k, col in enumerate(prob_columns(CLASSES)):
        tiles[col] = p[:, k]
    return tiles, p


class TestGridGeometry:
    def test_grid_shape_is_floor_division(self):
        # 1000 px at 0.5 um/px = 500 um; 500/32 = 15.6 -> 15 cells
        assert grid_shape((1000, 2000), 0.5, 32.0) == (31, 15)

    @pytest.mark.parametrize("mpp", [0.2201, 0.25, 0.5, 1.0, 2.0])
    def test_roundtrip_tile_centre_to_grid_cell(self, mpp):
        """Every tile lands in exactly one cell, and no two tiles collide.

        This is the inverse-mapping guarantee: floor(cx_px / step) must undo
        cx = (gx + 0.5) * step for every mpp the pipeline might use.
        """
        canvas = (4000, 3000)
        tissue = np.ones((300, 400), bool)
        tiles, _ = _tiles_from_grid(canvas, mpp, tissue)
        section = rasterize_tiles(tiles, canvas, mpp, CLASSES, grid_um=32.0)

        assert section.shape == grid_shape(canvas, mpp, 32.0)
        # One tile per occupied cell -- never two tiles averaged into one.
        assert section.tile_count.max() == 1
        assert int(section.tile_count.sum()) == len(tiles)

    def test_probabilities_land_at_the_right_cell(self):
        """A single tile's probabilities appear at its own grid cell only."""
        canvas, mpp = (2000, 2000), 0.5
        tissue = np.ones((200, 200), bool)
        tiles, p = _tiles_from_grid(canvas, mpp, tissue)

        keep = tiles.iloc[[7]].reset_index(drop=True)
        section = rasterize_tiles(keep, canvas, mpp, CLASSES, grid_um=32.0)

        step = 32.0 / mpp
        i = int(keep.cy_px[0] // step)
        j = int(keep.cx_px[0] // step)
        assert section.tile_count[i, j] == 1
        np.testing.assert_allclose(section.probs[:, i, j], p[7], rtol=1e-6)
        assert np.isnan(section.probs[:, :, :]).sum() == (
            section.probs.size - len(CLASSES)
        )


class TestSharedCanvas:
    def test_same_canvas_gives_same_shape_despite_different_tissue(self):
        """The property build_niche_volume's np.stack depends on."""
        canvas, mpp = (3000, 2500), 0.5

        sparse = np.zeros((250, 300), bool)
        sparse[100:120, 100:120] = True
        dense = np.ones((250, 300), bool)

        shapes = []
        for tissue in (sparse, dense):
            tiles, _ = _tiles_from_grid(canvas, mpp, tissue)
            shapes.append(
                rasterize_tiles(tiles, canvas, mpp, CLASSES, grid_um=32.0).shape
            )
        assert shapes[0] == shapes[1]

    def test_off_tissue_is_nan_not_zero(self):
        """NaN, not 0 -- a 0 probability is a real prediction, absence is not."""
        canvas, mpp = (2000, 2000), 0.5
        tissue = np.zeros((200, 200), bool)
        tissue[50:70, 50:70] = True
        tiles, _ = _tiles_from_grid(canvas, mpp, tissue)

        section = rasterize_tiles(tiles, canvas, mpp, CLASSES, grid_um=32.0)
        empty = section.tile_count == 0
        assert empty.any()
        assert np.isnan(section.probs[:, empty]).all()
        assert np.isfinite(section.probs[:, ~empty]).all()


class TestCoarsening:
    def test_coarser_grid_averages_tiles(self):
        """grid_um > tile_um aggregates by mean, reproducing 250 um reporting."""
        canvas, mpp = (4000, 4000), 0.5
        tissue = np.ones((400, 400), bool)
        tiles, _ = _tiles_from_grid(canvas, mpp, tissue)

        fine = rasterize_tiles(tiles, canvas, mpp, CLASSES, grid_um=32.0)
        coarse = rasterize_tiles(tiles, canvas, mpp, CLASSES, grid_um=256.0)

        assert coarse.shape[0] < fine.shape[0]
        assert coarse.tile_count.max() > 1
        assert int(coarse.tile_count.sum()) == len(tiles)
        # A mean of probabilities stays a probability.
        finite = np.isfinite(coarse.probs)
        assert (coarse.probs[finite] >= 0).all()
        assert (coarse.probs[finite] <= 1).all()


class TestArgmaxLabels:
    def test_label_index_comes_from_name_not_position(self):
        """classes_ is alphabetical; indices must not follow that order."""
        probs = np.full((4, 1, 4), 0.1, np.float32)
        for k in range(4):
            probs[k, 0, k] = 0.9  # class k wins at column k

        labels = argmax_labels(probs, CLASSES)
        expected = [NICHE_LABEL_INDEX[c] for c in CLASSES]
        assert list(labels[0]) == expected
        # epithelium is channel 1 (alphabetical) but label index 1, immune is
        # channel 2 but index 2 -- and acellular is channel 0, index 4.
        assert labels[0, 0] == NICHE_LABEL_INDEX["acellular"] == 4

    def test_all_nan_becomes_background(self):
        probs = np.full((4, 3, 3), np.nan, np.float32)
        probs[:, 1, 1] = [0.1, 0.7, 0.1, 0.1]
        labels = argmax_labels(probs, CLASSES)
        assert labels[1, 1] == NICHE_LABEL_INDEX["epithelium"]
        assert (labels[labels != NICHE_LABEL_INDEX["epithelium"]]
                == NICHE_BACKGROUND_INDEX).all()

    def test_unknown_class_raises(self):
        probs = np.zeros((1, 2, 2), np.float32)
        with pytest.raises(KeyError, match="No label index"):
            argmax_labels(probs, ["not_a_niche"])


class TestSmoothing:
    def test_nan_pattern_is_preserved(self):
        rng = np.random.default_rng(1)
        probs = rng.random((4, 20, 20)).astype(np.float32)
        probs[:, :5, :] = np.nan

        out = smooth_probs(probs, 1.5)
        assert np.isnan(out[:, :5, :]).all()
        assert np.isfinite(out[:, 5:, :]).all()

    def test_nans_do_not_drag_edge_values_down(self):
        """Normalised convolution: a constant field stays constant at the edge."""
        probs = np.full((1, 10, 10), 0.8, np.float32)
        probs[:, :3, :] = np.nan

        out = smooth_probs(probs, 2.0)
        np.testing.assert_allclose(out[0, 3:, :], 0.8, rtol=1e-5)

    def test_zero_sigma_is_a_copy(self):
        probs = np.full((2, 4, 4), 0.5, np.float32)
        out = smooth_probs(probs, 0.0)
        np.testing.assert_array_equal(out, probs)
        assert out is not probs


class TestValidation:
    def test_missing_probability_column_raises(self):
        tiles = pd.DataFrame({"cx_px": [10], "cy_px": [10], "p_immune": [0.5]})
        with pytest.raises(KeyError, match="missing required column"):
            rasterize_tiles(tiles, (1000, 1000), 0.5, CLASSES)

    @pytest.mark.parametrize("mpp", [0.0, -1.0])
    def test_bad_mpp_raises(self, mpp):
        tiles = pd.DataFrame(
            {"cx_px": [10], "cy_px": [10], **{c: [0.25] for c in prob_columns(CLASSES)}}
        )
        with pytest.raises(ValueError, match="mpp must be > 0"):
            rasterize_tiles(tiles, (1000, 1000), mpp, CLASSES)

    def test_grid_larger_than_canvas_raises(self):
        tiles = pd.DataFrame(
            {"cx_px": [10], "cy_px": [10], **{c: [0.25] for c in prob_columns(CLASSES)}}
        )
        with pytest.raises(ValueError, match="larger than the whole canvas"):
            rasterize_tiles(tiles, (100, 100), 0.5, CLASSES, grid_um=1000.0)
