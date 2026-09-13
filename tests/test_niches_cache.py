"""Tests for embedding checkpointing and the preview stride.

The embeddings are the only expensive part of the pipeline -- a 25 000-tile
section is over an hour of UNI2-h, while the classifier on top runs in
milliseconds. Caching them makes an interrupted section resume, and lets the
classifier be re-run without paying for the encoder again.

The dangerous failure is not a missing cache but a *wrong* one: silently
reusing embeddings computed for a different tile set, resolution or window
size would corrupt every downstream number with no error. The fingerprint
tests below are the guard on that.

A stub encoder stands in for UNI2-h so these run in milliseconds with no GPU
and no gated download.
"""

from __future__ import annotations

import numpy as np
import pytest

from path3d.niches.predict import (
    _EMBED_DIM,
    embed_tiles,
    grid_shape,
    load_embedding_cache,
    tile_fingerprint,
    tile_grid,
)


class _StubEncoder:
    """Deterministic stand-in for UNI2-h that counts how many tiles it saw."""

    def __init__(self):
        self.n_calls = 0
        self.n_tiles = 0

    def __call__(self, x):
        self.n_calls += 1
        self.n_tiles += int(x.shape[0])
        # A deterministic function of the input, so a resumed run and a
        # single-shot run produce byte-identical embeddings. Kept small
        # enough to survive the float16 cast the real path applies.
        seed = x.reshape(x.shape[0], -1).float().mean(dim=1, keepdim=True)
        return seed.expand(x.shape[0], _EMBED_DIM).clone()


@pytest.fixture
def encoder():
    import torch

    return (_StubEncoder(), "cpu", torch.float32)


@pytest.fixture
def scene():
    """A small synthetic slide plus the tile table over it."""
    rng = np.random.default_rng(0)
    h0, w0 = 2400, 2400
    level0 = rng.integers(0, 255, (h0, w0, 3), dtype=np.uint8)
    tissue = np.ones((240, 240), bool)
    tiles = tile_grid((w0, h0), 1.0, tissue, tile_um=200.0, fov_um=400.0)
    return level0, tiles


class TestStride:
    def test_stride_reduces_tile_count_quadratically(self):
        canvas, mpp = (8000, 6000), 0.5
        tissue = np.ones((600, 800), bool)

        full = tile_grid(canvas, mpp, tissue, tile_um=32.0, fov_um=112.0)
        strided = tile_grid(
            canvas, mpp, tissue, tile_um=32.0, fov_um=112.0, stride=4
        )
        assert len(strided) == pytest.approx(len(full) / 16, rel=0.05)

    def test_strided_tiles_stay_on_the_full_grid(self):
        """A preview must rasterise into the same cells as a full run."""
        from path3d.niches.rasterize import rasterize_tiles

        canvas, mpp, classes = (8000, 6000), 0.5, ["a", "b"]
        tissue = np.ones((600, 800), bool)

        full = tile_grid(canvas, mpp, tissue, tile_um=32.0, fov_um=112.0)
        strided = tile_grid(
            canvas, mpp, tissue, tile_um=32.0, fov_um=112.0, stride=4
        )
        # Every strided tile centre is also a full-run tile centre.
        full_centres = set(zip(full.cx_px, full.cy_px))
        assert set(zip(strided.cx_px, strided.cy_px)) <= full_centres

        for frame in (full, strided):
            for c in classes:
                frame[f"p_{c}"] = 0.5
        shapes = {
            len(frame): rasterize_tiles(
                frame, canvas, mpp, classes, grid_um=32.0
            ).shape
            for frame in (full, strided)
        }
        # Same raster shape, different fill density.
        assert len(set(shapes.values())) == 1
        assert next(iter(shapes.values())) == grid_shape(canvas, mpp, 32.0)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_bad_stride_raises(self, bad):
        with pytest.raises(ValueError, match="stride must be >= 1"):
            tile_grid((1000, 1000), 0.5, np.ones((10, 10), bool), stride=bad)


class TestFingerprint:
    def test_same_tiles_same_fingerprint(self, scene):
        _, tiles = scene
        assert tile_fingerprint(tiles, 1.0, 400.0) == tile_fingerprint(
            tiles.copy(), 1.0, 400.0
        )

    def test_fingerprint_tracks_mpp_and_window(self, scene):
        _, tiles = scene
        base = tile_fingerprint(tiles, 1.0, 400.0)
        assert tile_fingerprint(tiles, 0.5, 400.0) != base
        assert tile_fingerprint(tiles, 1.0, 224.0) != base

    def test_fingerprint_tracks_tile_positions(self, scene):
        _, tiles = scene
        moved = tiles.copy()
        moved.loc[0, "cx_px"] = int(moved.loc[0, "cx_px"]) + 1
        assert tile_fingerprint(moved, 1.0, 400.0) != tile_fingerprint(
            tiles, 1.0, 400.0
        )


class TestEmbeddingCache:
    def test_cache_is_written_and_complete(self, scene, encoder, tmp_path):
        level0, tiles = scene
        cache = tmp_path / "embeddings.npz"

        emb = embed_tiles(
            level0, tiles, mpp=1.0, fov_um=400.0, encoder=encoder,
            cache_path=cache, flush_every=8, verbose=False,
        )

        assert cache.exists()
        cached_emb, done, fp = load_embedding_cache(cache)
        assert done.all()
        assert fp == tile_fingerprint(tiles, 1.0, 400.0)
        np.testing.assert_array_equal(cached_emb, emb)

    def test_second_run_recomputes_nothing(self, scene, encoder, tmp_path):
        level0, tiles = scene
        cache = tmp_path / "embeddings.npz"

        first = embed_tiles(
            level0, tiles, mpp=1.0, fov_um=400.0, encoder=encoder,
            cache_path=cache, flush_every=8, verbose=False,
        )
        assert encoder[0].n_tiles == len(tiles)

        second = embed_tiles(
            level0, tiles, mpp=1.0, fov_um=400.0, encoder=encoder,
            cache_path=cache, flush_every=8, verbose=False,
        )
        # Not one extra tile through the encoder.
        assert encoder[0].n_tiles == len(tiles)
        np.testing.assert_array_equal(first, second)

    def test_resume_matches_an_uninterrupted_run(self, scene, encoder, tmp_path):
        """A run killed part-way and resumed must equal one that never stopped.

        Simulates the kill by truncating the cache to the first half.
        """
        level0, tiles = scene
        cache = tmp_path / "embeddings.npz"

        complete = embed_tiles(
            level0, tiles, mpp=1.0, fov_um=400.0, encoder=encoder,
            cache_path=cache, flush_every=8, verbose=False,
        )

        emb, done, fp = load_embedding_cache(cache)
        half = len(tiles) // 2
        partial_done = done.copy()
        partial_done[half:] = False
        partial_emb = emb.copy()
        partial_emb[half:] = 0
        np.savez(cache, emb=partial_emb, done=partial_done, fingerprint=np.array(fp))

        fresh = _StubEncoder()
        import torch

        resumed = embed_tiles(
            level0, tiles, mpp=1.0, fov_um=400.0,
            encoder=(fresh, "cpu", torch.float32),
            cache_path=cache, flush_every=8, verbose=False,
        )

        assert fresh.n_tiles == len(tiles) - half, "recomputed already-cached tiles"
        np.testing.assert_array_equal(resumed, complete)

    def test_stale_cache_is_never_partially_reused(self, scene, encoder, tmp_path):
        """The failure that would corrupt results silently.

        A cache written for a different resolution must be discarded whole,
        not merged into the new run.
        """
        level0, tiles = scene
        cache = tmp_path / "embeddings.npz"

        embed_tiles(
            level0, tiles, mpp=1.0, fov_um=400.0, encoder=encoder,
            cache_path=cache, flush_every=8, verbose=False,
        )
        stale = load_embedding_cache(cache)[0].copy()

        fresh = _StubEncoder()
        import torch

        # Same tiles, but claim a different window size -> different fingerprint.
        out = embed_tiles(
            level0, tiles, mpp=1.0, fov_um=200.0,
            encoder=(fresh, "cpu", torch.float32),
            cache_path=cache, flush_every=8, verbose=False,
        )
        assert fresh.n_tiles == len(tiles), "stale cache was reused"
        assert not np.array_equal(out, stale)
        assert load_embedding_cache(cache)[2] == tile_fingerprint(tiles, 1.0, 200.0)

    def test_corrupt_cache_is_ignored_not_fatal(self, scene, encoder, tmp_path):
        level0, tiles = scene
        cache = tmp_path / "embeddings.npz"
        cache.write_bytes(b"not an npz file")

        emb = embed_tiles(
            level0, tiles, mpp=1.0, fov_um=400.0, encoder=encoder,
            cache_path=cache, flush_every=8, verbose=False,
        )
        assert emb.shape == (len(tiles), _EMBED_DIM)
        assert load_embedding_cache(cache)[1].all()

    def test_missing_cache_returns_none(self, tmp_path):
        assert load_embedding_cache(tmp_path / "nope.npz") is None

    def test_no_cache_path_still_works(self, scene, encoder, tmp_path):
        level0, tiles = scene
        emb = embed_tiles(
            level0, tiles, mpp=1.0, fov_um=400.0, encoder=encoder, verbose=False
        )
        assert emb.shape == (len(tiles), _EMBED_DIM)
        assert not list(tmp_path.iterdir())


class TestCacheReuseForReclassification:
    def test_cached_embeddings_feed_the_classifier_directly(
        self, scene, encoder, tmp_path
    ):
        """The payoff: re-score a slide without re-running UNI2-h."""
        from path3d.niches.predict import classify_embeddings, load_model

        level0, tiles = scene
        cache = tmp_path / "embeddings.npz"
        embed_tiles(
            level0, tiles, mpp=1.0, fov_um=400.0, encoder=encoder,
            cache_path=cache, flush_every=8, verbose=False,
        )

        emb, done, _ = load_embedding_cache(cache)
        model = load_model()

        before = encoder[0].n_tiles
        _, loose = classify_embeddings(emb[done], model, quantile=0.5)
        _, tight = classify_embeddings(emb[done], model, quantile=0.95)
        assert encoder[0].n_tiles == before, "re-scoring must not touch the encoder"

        from path3d.niches.predict import call_columns

        for name in call_columns(model.classes):
            assert int(loose[name].sum()) >= int(tight[name].sum())
