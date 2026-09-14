"""Tests for continuous-probability quantification.

The property that matters: a soft volume must recover the true expected
volume where an argmax cannot. These tests construct volumes whose correct
answer is known analytically, so the argmax bias shows up as a number rather
than an opinion.
"""

from __future__ import annotations

import numpy as np
import pytest

from path3d.niches.quantify import (
    add_probability_labels,
    per_section_profile,
    probability_mask,
    soft_volumetrics,
    voxel_um_zyx,
)
from path3d.niches.rasterize import NicheSection
from path3d.niches.volume import build_niche_volume

pytest.importorskip("spatialdata")

CLASSES = ["acellular", "epithelium", "immune", "stroma"]


def _manifest(tmp_path, n):
    rows = ["section_index,filename,path"]
    paths = []
    for i in range(n):
        p = tmp_path / f"{i:04d}.ome.tiff"
        p.touch()
        paths.append(p)
        rows.append(f"{i},{p.name},{p}")
    csv = tmp_path / "manifest.csv"
    csv.write_text("\n".join(rows) + "\n")
    return csv, paths


def _volume(tmp_path, probs_per_section, shape=(10, 10), grid_um=32.0, z_um=12.0):
    csv, paths = _manifest(tmp_path, len(probs_per_section))
    sections = {}
    for p, per_class in zip(paths, probs_per_section):
        probs = np.zeros((len(CLASSES), *shape), np.float32)
        for k, value in enumerate(per_class):
            probs[k] = value
        sections[str(p)] = NicheSection(
            probs=probs,
            classes=list(CLASSES),
            tile_count=np.ones(shape, np.int32),
            grid_um=grid_um,
            canvas_wh=(shape[1] * 64, shape[0] * 64),
            mpp=0.5,
        )
    return build_niche_volume(csv, sections, z_spacing_um=z_um)


class TestVoxelSize:
    def test_reads_anisotropic_voxel_from_affine(self, tmp_path):
        sdata = _volume(tmp_path, [[0.25] * 4], z_um=12.0, grid_um=32.0)
        assert voxel_um_zyx(sdata) == pytest.approx((12.0, 32.0, 32.0))


class TestSoftVolumetrics:
    def test_recovers_the_analytic_expected_volume(self, tmp_path):
        """3 sections x 100 voxels at p=0.30 epithelium, 12x32x32 um voxels."""
        probs = [0.4, 0.3, 0.1, 0.2]
        sdata = _volume(tmp_path, [probs] * 3, shape=(10, 10))

        df = soft_volumetrics(sdata).set_index("class_name")
        voxel_um3 = 12.0 * 32.0 * 32.0
        expected = 0.30 * 300 * voxel_um3

        assert df.loc["epithelium", "expected_volume_um3"] == pytest.approx(expected)
        assert df.loc["epithelium", "mean_prob"] == pytest.approx(0.30)
        assert df.loc["epithelium", "volume_fraction"] == pytest.approx(0.30)

    def test_soft_fractions_sum_to_one(self, tmp_path):
        sdata = _volume(tmp_path, [[0.4, 0.3, 0.1, 0.2]] * 2)
        df = soft_volumetrics(sdata)
        assert df["volume_fraction"].sum() == pytest.approx(1.0)

    def test_argmax_collapses_a_uniform_volume_onto_one_class(self, tmp_path):
        """The bias, made numeric.

        Every voxel is 40% acellular / 30% epithelium. The argmax gives
        acellular 100% of the volume and epithelium none; the soft estimate
        recovers the real 40/30 split.
        """
        sdata = _volume(tmp_path, [[0.4, 0.3, 0.1, 0.2]] * 2)
        df = soft_volumetrics(sdata).set_index("class_name")

        assert df.loc["acellular", "argmax_volume_fraction"] == pytest.approx(1.0)
        assert df.loc["epithelium", "argmax_volume_fraction"] == pytest.approx(0.0)

        assert df.loc["acellular", "volume_fraction"] == pytest.approx(0.4)
        assert df.loc["epithelium", "volume_fraction"] == pytest.approx(0.3)

        # The dominant class is inflated, the minority erased.
        assert df.loc["acellular", "soft_over_argmax"] == pytest.approx(0.4)
        assert df.loc["epithelium", "soft_over_argmax"] == float("inf")

    def test_off_tissue_voxels_are_excluded_from_fractions(self, tmp_path):
        csv, paths = _manifest(tmp_path, 1)
        probs = np.full((len(CLASSES), 10, 10), np.nan, np.float32)
        probs[:, :5, :] = np.array([0.4, 0.3, 0.1, 0.2])[:, None, None]
        counts = np.zeros((10, 10), np.int32)
        counts[:5, :] = 1
        sections = {
            str(paths[0]): NicheSection(
                probs=probs, classes=list(CLASSES), tile_count=counts,
                grid_um=32.0, canvas_wh=(640, 640), mpp=0.5,
            )
        }
        sdata = build_niche_volume(csv, sections, z_spacing_um=12.0)

        df = soft_volumetrics(sdata).set_index("class_name")
        assert df.loc["epithelium", "n_tissue_voxels"] == 50
        # Fraction is over tissue, not over the whole padded canvas.
        assert df.loc["epithelium", "mean_prob"] == pytest.approx(0.3)

    def test_missing_probability_element_raises(self, tmp_path):
        sdata = _volume(tmp_path, [[0.25] * 4])
        del sdata.images["niche_probabilities"]
        with pytest.raises(KeyError, match="not found in sdata.images"):
            soft_volumetrics(sdata)

    def test_writes_csv(self, tmp_path):
        sdata = _volume(tmp_path, [[0.25] * 4])
        out = tmp_path / "vols.csv"
        soft_volumetrics(sdata, csv_path=out)
        assert out.exists()


class TestProbabilityMask:
    def test_per_section_threshold_adapts_to_each_slice(self, tmp_path):
        """A fixed global cutoff would empty the low-prevalence section.

        Section 0 has epithelium at 0.9, section 1 at 0.1. A global top-20%
        cutoff keeps only section 0; per-section keeps 20% of each.
        """
        csv, paths = _manifest(tmp_path, 2)
        sections = {}
        for p, level in zip(paths, (0.9, 0.1)):
            probs = np.full((len(CLASSES), 10, 10), 0.05, np.float32)
            probs[CLASSES.index("epithelium")] = level
            # a clear top corner so the quantile has something to select
            probs[CLASSES.index("epithelium"), :2, :2] = level + 0.05
            sections[str(p)] = NicheSection(
                probs=probs, classes=list(CLASSES),
                tile_count=np.ones((10, 10), np.int32), grid_um=32.0,
                canvas_wh=(640, 640), mpp=0.5,
            )
        sdata = build_niche_volume(csv, sections, z_spacing_um=12.0)

        per_sec = probability_mask(sdata, "epithelium", per_section=True)
        glob = probability_mask(sdata, "epithelium", per_section=False)

        assert per_sec[0].any() and per_sec[1].any(), "both sections keep voxels"
        assert glob[0].any() and not glob[1].any(), "global cutoff drops section 1"

    def test_off_tissue_never_selected(self, tmp_path):
        csv, paths = _manifest(tmp_path, 1)
        probs = np.full((len(CLASSES), 10, 10), np.nan, np.float32)
        probs[:, :5, :] = 0.25
        probs[CLASSES.index("epithelium"), :5, :] = np.linspace(
            0, 1, 50
        ).reshape(5, 10)
        counts = np.zeros((10, 10), np.int32)
        counts[:5, :] = 1
        sections = {
            str(paths[0]): NicheSection(
                probs=probs, classes=list(CLASSES), tile_count=counts,
                grid_um=32.0, canvas_wh=(640, 640), mpp=0.5,
            )
        }
        sdata = build_niche_volume(csv, sections, z_spacing_um=12.0)

        mask = probability_mask(sdata, "epithelium")
        assert not mask[:, 5:, :].any()

    def test_quantile_controls_how_much_is_kept(self, tmp_path):
        csv, paths = _manifest(tmp_path, 1)
        probs = np.zeros((len(CLASSES), 10, 10), np.float32)
        probs[CLASSES.index("epithelium")] = np.linspace(0, 1, 100).reshape(10, 10)
        sections = {
            str(paths[0]): NicheSection(
                probs=probs, classes=list(CLASSES),
                tile_count=np.ones((10, 10), np.int32), grid_um=32.0,
                canvas_wh=(640, 640), mpp=0.5,
            )
        }
        sdata = build_niche_volume(csv, sections, z_spacing_um=12.0)

        assert probability_mask(sdata, "epithelium", quantile=0.80).sum() == 20
        assert probability_mask(sdata, "epithelium", quantile=0.50).sum() == 50

    def test_unknown_class_raises(self, tmp_path):
        sdata = _volume(tmp_path, [[0.25] * 4])
        with pytest.raises(KeyError, match="is not a class of this volume"):
            probability_mask(sdata, "lymphocyte")

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_bad_quantile_raises(self, tmp_path, bad):
        sdata = _volume(tmp_path, [[0.25] * 4])
        with pytest.raises(ValueError, match=r"quantile must be in \[0, 1\)"):
            probability_mask(sdata, "epithelium", quantile=bad)


class TestAddProbabilityLabels:
    def test_adds_a_labels_element_with_the_same_geometry(self, tmp_path):
        sdata = _volume(tmp_path, [[0.4, 0.3, 0.1, 0.2]] * 2)
        key = add_probability_labels(sdata, "epithelium", quantile=0.5)

        assert key == "epithelium_call"
        element = sdata.labels[key]
        assert element.dims == ("z", "y", "x")
        assert np.asarray(element.data).dtype == np.uint8
        assert voxel_um_zyx(sdata, element_key=key) == pytest.approx(
            voxel_um_zyx(sdata)
        )

    def test_result_is_consumable_by_skimage(self, tmp_path):
        """The point of the bridge: connected components without the argmax."""
        from skimage.measure import label

        csv, paths = _manifest(tmp_path, 3)
        sections = {}
        for p in paths:
            probs = np.full((len(CLASSES), 10, 10), 0.2, np.float32)
            # Two separated high-probability blobs, so the component count is
            # a known quantity rather than an artefact of a tied field.
            probs[CLASSES.index("epithelium")] = 0.1
            probs[CLASSES.index("epithelium"), 1:3, 1:3] = 0.9
            probs[CLASSES.index("epithelium"), 7:9, 7:9] = 0.9
            sections[str(p)] = NicheSection(
                probs=probs, classes=list(CLASSES),
                tile_count=np.ones((10, 10), np.int32), grid_um=32.0,
                canvas_wh=(640, 640), mpp=0.5,
            )
        sdata = build_niche_volume(csv, sections, z_spacing_um=12.0)

        key = add_probability_labels(sdata, "epithelium", quantile=0.9)
        mask = np.asarray(sdata.labels[key].data)
        _, n = label(mask, connectivity=1, return_num=True)
        assert mask.sum() > 0
        # Two blobs per section, each continuous through all 3 sections.
        assert n == 2

    def test_uniform_field_yields_an_empty_mask(self, tmp_path):
        """A perfectly tied field has no 'top 20%' -- strict > keeps nothing.

        Matches predict_niches.py's own `P[:, i] > thr_c`. Worth pinning: a
        caller thresholding a flat region gets an empty compartment, not an
        arbitrary half of it.
        """
        sdata = _volume(tmp_path, [[0.4, 0.3, 0.1, 0.2]] * 2)
        assert probability_mask(sdata, "epithelium", quantile=0.5).sum() == 0


class TestPerSectionProfile:
    def test_one_row_per_section_and_class(self, tmp_path):
        sdata = _volume(tmp_path, [[0.4, 0.3, 0.1, 0.2]] * 3)
        df = per_section_profile(sdata)
        assert len(df) == 3 * len(CLASSES)
        assert sorted(df["section_index"].unique()) == [0, 1, 2]

    def test_z_um_uses_the_physical_spacing(self, tmp_path):
        sdata = _volume(tmp_path, [[0.25] * 4] * 3, z_um=12.0)
        df = per_section_profile(sdata)
        assert sorted(df["z_um"].unique()) == pytest.approx([0.0, 12.0, 24.0])

    def test_area_matches_the_soft_volume(self, tmp_path):
        """Per-section areas x section spacing must equal the total volume."""
        sdata = _volume(tmp_path, [[0.4, 0.3, 0.1, 0.2]] * 3)
        profile = per_section_profile(sdata)
        totals = soft_volumetrics(sdata).set_index("class_name")

        epi = profile[profile.class_name == "epithelium"]
        volume_mm3 = epi["expected_area_mm2"].sum() * (12.0 / 1000.0)
        assert volume_mm3 == pytest.approx(
            totals.loc["epithelium", "expected_volume_mm3"]
        )
