# Xenium tissue neighbourhoods from H&E

Predicts, from a routine H&E whole-slide image alone, a four-class tissue-neighbourhood map:

| class | what it is | source Xenium niches |
|---|---|---|
| **epithelium** | secretory tumour epithelium (all secretory niches merged) | nb_1, nb_2, nb_5, nb_6, nb_7 |
| **immune** | immune-rich neighbourhood | nb_4 |
| **stroma** | fibroblast / vascular / mesothelial | nb_3, nb_8, nb_9, nb_10 |
| **acellular** | necrosis, dense ECM, mucin | acellular |

No spatial transcriptomics needed at inference. Labels originate from Xenium-derived 50 µm
neighbourhoods on 8 HGSC whole tumours (Cook Lab, `2026_final_xenium_analysis`).
Contact: snersesian@ohri.ca

**Why the epithelial niches are merged:** the individual secretory niches encode the
SecA→transitioning→SecB polarisation axis, and that axis is **not** recoverable from morphology.
We tested it extensively (per-class F1 0.03–0.17, unchanged at 14× the training data; a separate
continuous SecB model reached only AUC 0.65–0.78 with no reliability estimate). Collapsed into one
epithelium class, the compartment is read reliably. That trade is the point of this model.

---

## Validation

**Leave-one-TUMOUR-out over 8 whole tumours** — every score below is from a model that never saw
that tumour. Scored one-vs-rest by AUC, which is prevalence-independent.

| class | AUC mean | AUC worst tumour | correctly labelled (confusion) | prevalence range |
|---|---|---|---|---|
| **immune** | **0.923** | 0.804 | 75% | 1.2–37.5% |
| **epithelium** | **0.911** | 0.845 | 82% | 4.5–98.4% |
| stroma | 0.869 | 0.626 | 56% | 0.7–45.0% |
| acellular | 0.866 | 0.814 | 63% | 0.6–20.2% |

**Epithelium and immune are the dependable classes.** Stroma is good on average but drops to
AUC 0.626 on one tumour (an unusual case: only 31% epithelium, mostly stroma and immune), and 17%
of true stroma is labelled immune — biologically unsurprising, since both are non-epithelial
cellular tissue. Treat stroma and acellular as regional tendencies rather than firm calls.

### Read AUC, and threshold per slide
Prevalence varied enormously between our tumours (epithelium 4.5%–98.4%). A fixed global `argmax`
therefore under-calls rare classes even when ranking is excellent: on our 4.5%-epithelium tumour
the epithelium AUC was 0.935 while its F1 was 0.23. The script writes both an `argmax_class` and
per-class **per-slide** top-quantile calls (`call_*`, `--quantile`, default top 20%). Use the
continuous `p_*` columns wherever you can, and the per-slide calls when you need a binary mask.

---

## Install

```bash
pip install "timm>=0.9.8" huggingface_hub torch tifffile zarr scikit-image scipy joblib pillow matplotlib
```

**Hugging Face access is required** — the feature extractor (`MahmoodLab/UNI2-h`) is gated:

1. Request access at <https://huggingface.co/MahmoodLab/UNI2-h> (institutional email; approval is
   manual but not slow)
2. `hf auth login`, pasting a read token from <https://huggingface.co/settings/tokens>
   (`huggingface_hub` 1.x renamed the CLI — it is `hf auth login`, **not** `huggingface-cli login`)

First run downloads ~2.5 GB of weights and caches them.

## Run

```bash
python predict_niches.py SLIDE.ome.tif --mpp 0.2201 --out results_slide1
```

### `--mpp` is required and must be your own
It is the microns-per-pixel of **your** level-0 image. The model works in microns (32 µm grid,
112 µm windows), so a wrong value silently changes the physical size of every tile and degrades
everything. Read it from your scanner metadata. Ours was 0.2201 — **do not assume yours matches.**

### Outputs
| file | contents |
|---|---|
| `tiles_niches.csv` | per 32 µm tile: `cx_px, cy_px, p_epithelium, p_immune, p_stroma, p_acellular, argmax_class, call_*` |
| `niche_maps_250um.npz` | one 250 µm probability grid per class + `tile_count` (NaN off-tissue) |
| `niche_maps.png` | H&E, most-likely-niche map, and one probability map per class |
| `summary.json` | tile count, tissue area, mean probability and argmax share per class |

Runtime is dominated by the feature extractor: ~17 tiles/s on an Apple M3 Max, faster on CUDA. A
60 mm² section is ~60,000 tiles ≈ 1 hour. VRAM ≥8 GB is ample.

---

## For stacking serial sections

Report at **250 µm**, not per tile — that was the optimum of a proper sweep (finer is noisy,
coarser over-smooths), and per-tile agreement was substantially weaker than per-region.

Tile coordinates are in each slide's own level-0 pixels, so **sections are not co-registered to one
another** — align them yourself. The `niche_maps_250um.npz` grids are reasonable registration
targets, being smooth and tissue-shaped.

---

## What has and has not been tested

**Tested.** The four-class model is validated as above. The script's full path — pyramidal TIFF
reading, block-wise level-0 crops, tissue detection, Macenko normalisation, UNI2-h features, model,
250 µm aggregation, figures — was executed end-to-end on a synthetic slide. The tissue detector was
checked directly: it kept 100% of stained tissue, and rejected 0% / 0% of a grey ink mark and pale
background respectively.

**Not tested.** The script has never run on a real slide from a different scanner, because the
original whole-slide images were not on our volume when the package was assembled. The image-reading
code is lifted unchanged from the pipeline that processed 260,000 real tiles, so the risk is low —
but **run one slide first and look at `niche_maps.png` before trusting a batch.**

## Limits

**HGSC only**, post-Xenium H&E, one scanner, 0.2201 µm/px. Other tumour types, stains or scanners
are extrapolation. Macenko normalisation to a fixed reference is applied automatically and helps
with stain differences, but is not a guarantee.

**Tissue detection is image-only** (dark *and* saturated pixels, small islands dropped) because
you have no cell positions to define tissue. Untested against pen marks, large air bubbles, or
heavy artefact.

**This is a compartment map, not a cell-state map.** It does not resolve SecA / transitioning /
SecB within the epithelium — see the note at the top.

**No per-slide reliability estimate.** All 8 tumours gave epithelium AUC ≥ 0.845, so the risk is
low, but we cannot certify a *new* slide in advance. The most predictive warning sign we found was
distance from the training distribution in feature space (Spearman −0.67 against accuracy, n=8,
p=0.07) — suggestive, not established, and not implemented here.

**Sanity-check once per tissue block.** Even a pathologist's eye on the `niche_maps.png` overlay
converts an unquantified risk into a known one.
