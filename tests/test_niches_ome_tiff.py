"""Integration test against one real registered OME-TIFF.

Run it with the path supplied at runtime -- never hardcoded, because this is a
public repository and the slides are patient-derived::

    pytest tests/test_niches_ome_tiff.py --ome-tiff /path/to/0000.ome.tiff

    # or
    PATH3D_TEST_OME_TIFF=/path/to/0000.ome.tiff pytest tests/test_niches_ome_tiff.py

Without a path the whole module skips, so a clean checkout with no data still
runs green.

What this covers is everything between the registered file and the 3D volume:
metadata, pyramid, tissue detection, tile grid, rasterisation and volume
assembly. It deliberately does **not** require the UNI2-h encoder -- that is
gated, 2.5 GB, and slow. Add ``--run-encoder`` to also push a handful of real
tiles through the actual model.
"""

from __future__ import annotations

import numpy as np
import pytest

from path3d.niches.predict import (
    FOV_UM,
    canvas_wh,
    grid_shape,
    load_model,
    prob_columns,
    read_mpp,
    tile_grid,
    tissue_mask,
)
from path3d.niches.rasterize import argmax_labels, rasterize_tiles

pytestmark = pytest.mark.slow

# Coarsest sensible level for tissue detection on a real section. The niche CLI
# defaults to 4; anything finer makes this test slow and memory-hungry.
_MASK_LEVEL = 4

# The resolution the niche model was trained at (model card: "Ours was 0.2201
# -- do not assume yours matches"). Used only to report how much detail a
# given stack's warp resolution is giving up, never to override a file's own
# metadata.
_TRAINED_MPP = 0.2201


@pytest.fixture(scope="module")
def slide(ome_tiff_path):
    """Metadata + a coarse thumbnail, read once for the whole module."""
    import tifffile

    with tifffile.TiffFile(str(ome_tiff_path)) as tf:
        series = tf.series[0]
        levels = series.levels
        n_levels = len(levels)
        level = min(_MASK_LEVEL, n_levels - 1)
        thumb = levels[level].asarray()

    return {
        "path": ome_tiff_path,
        "mpp": read_mpp(ome_tiff_path),
        "canvas": canvas_wh(ome_tiff_path),
        "n_levels": n_levels,
        "mask_level": level,
        "thumb": thumb,
    }


class TestMetadata:
    def test_mpp_is_readable_and_plausible(self, slide):
        """read_mpp must recover the resolution warp_and_save_section stamped.

        A wrong mpp silently rescales every tile, which is the single easiest
        way to get meaningless niche output -- so this is the check that keeps
        --mpp from ever needing to be typed by hand.
        """
        mpp = slide["mpp"]
        assert 0.05 < mpp < 5.0, (
            f"mpp {mpp} is outside any plausible brightfield WSI range; the "
            f"resolution metadata in this file is probably wrong."
        )

    def test_canvas_matches_level_zero(self, slide):
        import tifffile

        with tifffile.TiffFile(str(slide["path"])) as tf:
            h, w = tf.series[0].levels[0].shape[:2]
        assert slide["canvas"] == (w, h)

    def test_pyramid_is_deep_enough_for_tissue_detection(self, slide):
        """A shallow pyramid makes tissue detection read a near-full-res level.

        predict_slide degrades gracefully (it clamps mask_level and warns), but
        on a real section that means decoding gigapixels to find tissue.
        """
        assert slide["n_levels"] >= 2, (
            f"{slide['path'].name} has {slide['n_levels']} pyramid level(s). "
            f"registration.warp_and_save_section writes pyramid=True; a flat "
            f"file here means the warp was not saved pyramidally."
        )
        if slide["n_levels"] <= _MASK_LEVEL:
            pytest.xfail(
                f"pyramid has {slide['n_levels']} levels, fewer than the "
                f"default mask_level={_MASK_LEVEL}; tissue detection will run "
                f"at level {slide['mask_level']} and be slow on a full stack."
            )

    def test_section_is_a_plausible_physical_size(self, slide):
        w, h = slide["canvas"]
        mpp = slide["mpp"]
        width_mm, height_mm = w * mpp / 1000, h * mpp / 1000
        assert 1.0 < width_mm < 200.0
        assert 1.0 < height_mm < 200.0


class TestTissueDetection:
    def test_finds_some_but_not_all_tissue(self, slide):
        """A registered section is mostly canvas fill; tissue is the minority.

        The niche detector requires darkness AND saturation, which is what
        rejects VALIS's near-white warp padding -- a darkness-only threshold
        would keep it.
        """
        mask = tissue_mask(slide["thumb"])
        frac = float(mask.mean())
        assert 0.005 < frac < 0.95, (
            f"tissue fraction {frac:.3f} -- either nothing was detected "
            f"(check the file) or the canvas fill is being kept as tissue."
        )

    def test_canvas_fill_is_rejected(self, slide):
        """The brightest, least saturated pixels must not be called tissue."""
        rgb = slide["thumb"][..., :3].astype(np.float32)
        gray = rgb.mean(2)
        mx, mn = rgb.max(2), rgb.min(2)
        sat = np.divide(mx - mn, np.maximum(mx, 1e-6))

        fill = (gray > 240) & (sat < 0.05)
        if fill.sum() < 100:
            pytest.skip("no near-white unsaturated region in this section")
        assert not tissue_mask(slide["thumb"])[fill].any()


class TestTileGrid:
    def test_tiles_land_on_tissue_and_inside_the_canvas(self, slide):
        model = load_model()
        mask = tissue_mask(slide["thumb"])
        tiles = tile_grid(
            slide["canvas"],
            slide["mpp"],
            mask,
            tile_um=model.tile_um,
            fov_um=model.fov_um,
        )

        assert len(tiles) > 0, "no tiles on tissue -- check mpp and the slide"

        w, h = slide["canvas"]
        fov = int(round(model.fov_um / slide["mpp"]))
        assert (tiles.x0_px >= 0).all()
        assert (tiles.y0_px >= 0).all()
        assert (tiles.x0_px + fov <= w).all()
        assert (tiles.y0_px + fov <= h).all()

    def test_tile_count_matches_tissue_area(self, slide):
        """Each tile is tile_um^2 of tissue; the total should be sane for a section."""
        model = load_model()
        mask = tissue_mask(slide["thumb"])
        tiles = tile_grid(
            slide["canvas"], slide["mpp"], mask, tile_um=model.tile_um,
            fov_um=model.fov_um,
        )
        area_mm2 = len(tiles) * model.tile_um**2 / 1e6
        assert 0.05 < area_mm2 < 2000.0, f"{area_mm2:.2f} mm2 of tissue"

    def test_window_carries_the_detail_the_model_was_trained_on(self, slide):
        """The resolution trap, checked against the file rather than assumed.

        The model crops a 112 um window and resizes it to 224x224. It was
        trained at 0.2201 um/px, where that window is 509 px and is
        DOWNSAMPLED 2.3x into the network. A section registered at path3d's
        default ``cfg.SEG_MPP`` of 0.5 um/px gives a 224 px window -- the right
        physical size, but carrying roughly a fifth of the detail, with no
        resize happening at all.

        The physical size is asserted (that is a hard requirement). The
        effective magnification is an xfail, because it is a property of how
        the stack was warped, not a bug in this code.
        """
        model = load_model()
        fov_px = int(round(model.fov_um / slide["mpp"]))
        assert fov_px * slide["mpp"] == pytest.approx(FOV_UM, abs=slide["mpp"])

        trained_fov_px = model.fov_um / _TRAINED_MPP
        if fov_px < trained_fov_px / 1.5:
            pytest.xfail(
                f"a {model.fov_um:.0f} um window is {fov_px} px at "
                f"{slide['mpp']:.4f} um/px, against {trained_fov_px:.0f} px at "
                f"the training resolution of {_TRAINED_MPP} um/px "
                f"({trained_fov_px / fov_px:.1f}x less detail into a 224x224 "
                f"input). Re-warp this stack at the slide's native mpp for the "
                f"niche pass -- see docs/niches/INTEGRATION.md."
            )


class TestRasterisation:
    def test_grid_shape_is_derived_from_the_canvas(self, slide):
        """The property that lets sections with different tissue stack together."""
        model = load_model()
        mask = tissue_mask(slide["thumb"])
        tiles = tile_grid(
            slide["canvas"], slide["mpp"], mask, tile_um=model.tile_um,
            fov_um=model.fov_um,
        )
        rng = np.random.default_rng(0)
        p = rng.random((len(tiles), len(model.classes)))
        p /= p.sum(1, keepdims=True)
        for k, col in enumerate(prob_columns(model.classes)):
            tiles[col] = p[:, k]

        section = rasterize_tiles(
            tiles, slide["canvas"], slide["mpp"], model.classes,
            grid_um=model.tile_um,
        )

        assert section.shape == grid_shape(
            slide["canvas"], slide["mpp"], model.tile_um
        )
        assert int(section.tile_count.sum()) == len(tiles)
        assert 0.0 < section.tissue_frac < 1.0

    def test_labels_only_where_tiles_were_predicted(self, slide):
        model = load_model()
        mask = tissue_mask(slide["thumb"])
        tiles = tile_grid(
            slide["canvas"], slide["mpp"], mask, tile_um=model.tile_um,
            fov_um=model.fov_um,
        )
        for k, col in enumerate(prob_columns(model.classes)):
            tiles[col] = 0.9 if k == 1 else 0.1

        section = rasterize_tiles(
            tiles, slide["canvas"], slide["mpp"], model.classes,
            grid_um=model.tile_um,
        )
        labels = argmax_labels(section.probs, model.classes)

        predicted = section.tile_count > 0
        assert (labels[predicted] != 0).all()
        assert (labels[~predicted] == 0).all()


class TestVolumeFromOneSection:
    def test_single_section_volume_has_the_right_geometry(self, slide, tmp_path):
        """The real canvas all the way through to a SpatialData volume."""
        pytest.importorskip("spatialdata")
        from path3d.niches.volume import build_niche_volume

        model = load_model()
        mask = tissue_mask(slide["thumb"])
        tiles = tile_grid(
            slide["canvas"], slide["mpp"], mask, tile_um=model.tile_um,
            fov_um=model.fov_um,
        )
        for k, col in enumerate(prob_columns(model.classes)):
            tiles[col] = 0.7 if k == 1 else 0.1

        # A coarse grid keeps this test's memory bounded on a full-size
        # section; the geometry assertions are resolution-independent.
        grid_um = 250.0
        section = rasterize_tiles(
            tiles, slide["canvas"], slide["mpp"], model.classes, grid_um=grid_um
        )

        slide_path = tmp_path / "0000.ome.tiff"
        slide_path.touch()
        manifest = tmp_path / "manifest.csv"
        manifest.write_text(
            f"section_index,filename,path\n0,{slide_path.name},{slide_path}\n"
        )

        sdata = build_niche_volume(
            manifest, {str(slide_path): section}, z_spacing_um=12.0
        )

        labels = sdata.labels["tissue_labels"]
        probs = sdata.images["niche_probabilities"]
        assert labels.shape == (1, *section.shape)
        assert probs.shape == (len(model.classes), 1, *section.shape)

        from spatialdata.transformations import get_transformation

        m = get_transformation(labels, "microns_3d").to_affine_matrix(
            input_axes=("z", "y", "x"), output_axes=("z", "y", "x")
        )
        assert m[0, 0] == pytest.approx(12.0)
        assert m[1, 1] == pytest.approx(grid_um)

        # The volume's physical footprint must match the slide's, up to the
        # partial cell at each far edge. grid_shape floors, exactly as the
        # original predict_niches.py did, so the volume covers a whole number
        # of cells and never extends past the canvas.
        _, ny, nx = labels.shape
        w, h = slide["canvas"]
        width_um, height_um = w * slide["mpp"], h * slide["mpp"]
        assert 0 <= width_um - nx * grid_um < grid_um
        assert 0 <= height_um - ny * grid_um < grid_um


class TestRealEncoder:
    """Opt-in: the actual gated UNI2-h forward pass on a few real tiles."""

    def test_predicts_calibrated_probabilities(self, slide, run_encoder):
        if not run_encoder:
            pytest.skip("pass --run-encoder to exercise the gated UNI2-h model")

        from path3d.niches.predict import predict_slide

        tiles, meta = predict_slide(
            slide["path"], mask_level=slide["mask_level"], limit=8, verbose=False
        )

        assert len(tiles) == 8
        assert meta["mpp"] == pytest.approx(slide["mpp"])
        assert meta["canvas_wh"] == slide["canvas"]

        cols = prob_columns(meta["classes"])
        p = tiles[cols].to_numpy()
        assert np.isfinite(p).all()
        assert (p >= 0).all() and (p <= 1).all()
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)
        assert set(tiles["argmax_class"]) <= set(meta["classes"])
