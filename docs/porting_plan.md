# path3d: Python Porting Plan

> **Note (2026-04-29)**: this plan has been through two rounds of independent review (`plan_review.md` and `plan_review_2.md`). Where this document conflicts with either review, the reviews supersede. The most consequential change from review #2: **CODA functional parity is no longer the validation target** — see "Acceptance Criteria" below. Several Tier 1 algorithm choices have been updated accordingly. Items where the right choice depends on PI input are flagged in "Open Decisions Required" and should be resolved before the student starts the affected module.

## Design Principles

1. **Right tool for the lab's objective**, which is high-quality 3D reconstruction + compartment segmentation of HGSC / LGSC / endometriosis FFPE, with downstream Xenium / snPATHO-seq / IHC / mIF integration. CODA is a reference implementation, not the target. Deviate where a 2026 alternative is clearly better.
2. **Modular architecture** — each pipeline stage is an independent module with clean interfaces. Modules expose first-class outputs (transforms, manifests, multimodal volume containers) that downstream stages can consume without re-reading raw images.
3. **Validate against the lab's pilot tissue, not against MATLAB.** MATLAB CODA outputs on pancreas are at best a sanity check; they are not the deliverable's success criterion. See "Acceptance Criteria" below.
4. **Leverage modern, maintained libraries** — pathology foundation models, Cellpose-SAM, SpatialData — rather than reimplementing algorithms or anchoring on what CODA used in 2022.

## Acceptance Criteria

The Tier 1 + 1.5 deliverable is accepted when the following are true on the lab's pilot specimen:

1. **Registration TRE ≤ 25 µm** measured against ≥50 manually-placed fiducial landmarks (anatomically distinctive points: vessel branches, gland lumens) on at least 5 pairs of adjacent sections. ≤15 µm is the eventual goal for clean Xenium overlay (~1 cell-diameter); 25 µm is the Tier 1 acceptance threshold.
2. **Segmentation Dice ≥ 0.85** per major compartment on a held-out internal annotation set (HGSC, LGSC, or endometriosis depending on pilot tissue choice). Annotations produced in QuPath, exported as GeoJSON.
3. **Registered cell-coordinate stability**: the same nucleus, detected on adjacent sections, lands within ≤15 µm of itself in the registered 3D volume on ≥95% of paired detections (sampled on a manually verified set of ~100 nuclei).
4. **Reproducible end-to-end pipeline**: one command runs preprocessing → registration → segmentation → cell detection → volume → quantification on a manifest CSV and produces SpatialData and AnnData outputs deterministically.

CODA-output comparison (MATLAB on pancreas) is one *sanity check* among several, not a deliverable. Skip it entirely if MATLAB CODA is not accessible — see Open Decisions #6.

## Open Decisions Required

Before the student commits to specific architecture choices, the PI needs to resolve:

1. **GPU budget.** Foundation-model segmentation (UNI2 / H-optimus-0) and Cellpose-SAM both assume an A100-class GPU (≥40 GB). If the lab has only shared low-memory GPUs or no GPU access, fall back to ResNet50/MobileNet backbones and Cellpose v3.
2. **Annotation budget.** Foundation-model segmentation reaches Dice ≥ 0.85 with ~50–100 annotated 1024×1024 patches per compartment (~2 weeks of pathologist effort). If pathologist time is essentially zero, the Tier 1 plan needs SAM-prompted / weakly supervised annotation workflows (Path-SAM2, EP-SAM) — a meaningful design change.
3. **Pilot tissue selection.** PROJECT_SPEC lists HGSC dormancy block, LGSC rare-subtype block, or endometriosis resection. Reviewer #2 recommends HGSC dormancy block (most heterogeneous tissue → hardest validation; most informative Xenium overlay; directly serves a funded grant). PI's call.
4. **Commercialization stance.** Cellpose 4 is non-commercial; UNI2 is gated CC-BY-NC; H-optimus-0 is more permissive; VALIS is GPL-3. If path3d may eventually be released for industry collaboration or commercial use, several Tier 1 defaults change.
5. **path3d's own license.** Bundling VALIS forces GPL-3 on path3d. If a permissive license (BSD/MIT/Apache) is required, registration must be pluggable, not VALIS-only.
6. **MATLAB CODA access for sanity checks.** Do we have a MATLAB CODA install + a small reference dataset? If not, all "validate against MATLAB" items in this plan should be deleted (don't promise validation we can't perform).
7. **Forjaz / Crawford reproducibility check.** Does the pilot need to reproduce a specific Forjaz et al. or Crawford et al. analysis on lab tissue, as a publication-grade demonstration? If yes, this anchors Tier 2 features (lumen skeletonization, virtual re-sectioning, hotspot/coldspot) more firmly.

## Proposed Module Structure

```
path3d/
├── __init__.py          # Public API exports
├── slide_io.py          # WSI reading, metadata extraction, image export
├── preprocessing.py     # Downsampling, tissue masking, stain normalization
├── registration.py      # Rigid + elastic serial section registration
├── segmentation.py      # Model training and inference for tissue labeling
├── nuclear_detection.py # Nuclear detection from hematoxylin channel
├── volume.py            # 3D volume construction from registered 2D data
├── quantification.py    # Spatial statistics, morphometrics, volumetrics
├── visualization.py     # 3D rendering and plotting
├── pipeline.py          # End-to-end orchestration
├── config.py            # Tissue class definitions, colors, default parameters
└── utils.py             # Shared utilities (color deconvolution, etc.)
```

Note: Do NOT name the I/O module `io.py` — this shadows Python's built-in `io` module and causes import errors.

## Python Library Mapping

| Pipeline Stage | MATLAB Tool | Python Replacement | Notes |
|---|---|---|---|
| WSI I/O | Openslide (MEX) | `openslide-python` or `tiffslide` | `tiffslide` is a pure-Python drop-in; `cucim` for GPU |
| Downsampling | MATLAB `imresize` | `scikit-image` (`skimage.transform.resize`) | `order=0` for label maps; otherwise default |
| Tissue masking | Green channel + RGB std dev | `scikit-image` + `numpy` | CODA's dual-criterion is fine; Otsu as simpler alternative. The tissue mask must also gate segmentation inference (don't run the network on background) |
| Stain normalization | Custom k-means OD clustering | `torchstain` Macenko, **optional** | Tier 1 must include a "no normalization" path and benchmark it against `macenko` under the chosen segmentation backbone. Foundation-model encoders are partially robust to stain variance and may make normalization unnecessary or harmful |
| Color deconvolution | Custom MATLAB | `skimage.color.separate_stains` | Lives in `utils.py` (NOT `nuclear_detection.py`) — needed for IHC quantification too |
| Rigid registration | Radon transform + cross-correlation | Multiple candidates — see Tier 1.2 | **No default committed yet.** Validation harness must benchmark VALIS, DeeperHistReg, ANTs/SimpleElastix B-spline, ZeroReg3D, and a faithful Radon reimplementation before Tier 1 commits |
| **Full registration pipeline** | Custom scripts | **TBD via month-1 benchmark** | See Open Decisions #4–5 (license) and Tier 1.2 below |
| Segmentation model | DeepLab v3+ (DL Toolbox) | **Pathology FM encoder + decoder (primary)**; DeepLab v3+ as baseline | Primary: UNI2 (`MahmoodLab/UNI2-h`) or H-optimus-0 frozen + lightweight FPN/Mask2Former decoder. Baseline: `segmentation_models_pytorch` `DeepLabV3Plus(encoder_name='resnet50')`. Both implemented; FM is the production default if GPU + annotation budget allow |
| Segmentation training | DL Toolbox training loop | PyTorch training loop (frozen encoder + trainable decoder, optionally with LoRA on encoder) | Standard supervised training. Training data: ~50–100 annotated 1024×1024 patches per compartment, exported from QuPath as GeoJSON |
| Nuclear detection | Crocker-Grier particle tracking (`pkfndW`) | **Cellpose-SAM** (`cellpose>=4`, `cpsam` model) | Instance masks (not centroids) → morphometrics fall out for free; Tier 2 "Cellpose upgrade" collapses into Tier 1. `trackpy` kept available as a fallback for sanity-check benchmarks, NOT the default |
| 3D volume assembly | MATLAB matrix ops | **`SpatialData`-backed multimodal container** (or `xarray.Dataset` if `spatialdata` 3D support is insufficient) — see Tier 1.5 | Lazy zarr storage, explicit per-section `z_microns` coordinate, modalities as labeled variables. NOT a single `(Z,Y,X)` NumPy stack |
| 3D visualization | `isosurface` / `patch` | `pyvista`, `napari`, or `vedo` | `napari-spatialdata` for interactive inspection; `pyvista` for publication |
| Annotation I/O | Aperio XML | `geojson` (QuPath export) | Default to QuPath GeoJSON; XML is supported only for legacy inputs |

## Tiered Implementation Plan

### Tier 1: Essential (Core MVP)

These modules reproduce the core CODA pipeline. Complete these first.

#### 1.1 I/O and Preprocessing (`slide_io.py`, `preprocessing.py`)

**Replaces**: `create_downsampled_tif_images.m`, `calculate_tissue_ws.m`, `get_mpp_of_image.m`, `deconvolve_histological_images.m`

Key implementation notes:
- Use `openslide-python` to read .svs/.ndpi/.tiff WSIs. The `OpenSlide.read_region()` method reads arbitrary regions at any level. `slide.level_dimensions` gives available downsampled sizes. `slide.properties['openslide.mpp-x']` gives microns-per-pixel.
- **Important**: `read_region()` returns **RGBA** (4 channels). Always strip the alpha channel: `.convert("RGB")` or `np.array(tile)[:, :, :3]`.
- For tissue masking: implement CODA's dual-criterion approach — regions with **low green channel intensity** AND **high RGB standard deviation**. Apply morphological cleanup (`skimage.morphology.remove_small_objects`, `binary_fill_holes`). Offer Otsu as a simpler alternative.
- **Stain normalization** is implemented in Tier 1 but is **optional and benchmarked**. API: `path3d.preprocessing.normalize_stain(method='macenko'|'reinhard'|'none')`. Default determined by experiment, not by assumption: run segmentation with and without normalization on a held-out validation section; if Dice improves by ≥0.02 with normalization, default to `macenko`. Otherwise default to `none`. Foundation-model encoders trained on multi-institution data are substantially more robust to stain variance than ImageNet networks, and 2025 benchmarks have shown stain normalization can *reduce* FM embedding consistency in some cases. Reference image: pick a morphologically central, well-stained section (not the first one); document the choice; cache the Macenko stain matrix for reproducibility. Use `torchstain` (GPU); `histomicstk` as fallback.
- **Section QC**: After masking, report tissue area per section and flag outliers (anomalously small area = damaged section, anomalously high stain intensity = possible fold).
- Save intermediates as standard TIFF (via `tifffile`) or OME-TIFF for metadata-rich output.

```python
# Rough API sketch
import openslide

slide = openslide.OpenSlide("section_001.svs")
mpp = float(slide.properties['openslide.mpp-x'])
thumbnail = np.array(slide.get_thumbnail((2000, 2000)).convert("RGB"))
```

#### 1.2 Registration (`registration.py`)

**Replaces**: `calculate_image_registration.m`, `apply_image_registration.m`

**Approach: validation harness first, tool choice second.** Do not commit to a registration backend before benchmarking on real lab tissue. This is the highest-leverage de-risking step in Tier 1 — the rest of the pipeline is built on top of registered coordinates, and silently inadequate registration accuracy will not surface until cell coordinates fail to line up across z and Xenium overlay misaligns.

**Month 1 deliverable: registration validation harness.**
- 20–30 H&E sections from a real pilot block
- 50–100 manually placed fiducial landmarks paired across ≥5 pairs of adjacent sections (vessel branches, gland lumens, fat clusters)
- Automated TRE computation in microns
- Acceptance threshold: TRE ≤ 25 µm for Tier 1; ≤15 µm aspirational for Xenium overlay

**Tools to benchmark (run all on the pilot stack, report TRE):**

1. **VALIS** (`valis-wsi`, GPL-3) — feature-based (BRISK/ORB + RANSAC) + B-spline / thin-plate spline elastic; published TRE single-digit microns on ANHIR/ACROBAT. Caveats: requires libvips + JVM (Bio-Formats), JVM lifecycle requires `try/finally` with `registration.kill_jvm()` or pipelines hang on cluster nodes; license is GPL-3 (see Open Decisions #4–5).
2. **DeeperHistReg** (Wodzinski 2021, CMPB; actively maintained 2024–2025) — deep-learning rotation initialization + B-spline. Built specifically for differently-stained histology but works on H&E. ~3 sec/pair on GPU. Permissive license. No JVM.
3. **ANTs / SimpleElastix B-spline** (`SimpleITK`) — mature, well-tested, scriptable, no JVM, permissive license. Slower but a strong reliable baseline.
4. **ZeroReg3D** (arXiv 2025) — designed specifically for 3D consecutive histology. Worth a benchmark; maturity unclear.
5. **Faithful CODA reimplementation** — Radon-transform rotation finding (`skimage.transform.radon`) + `scipy.signal.fftconvolve` translation + multi-reference strategy + tiled elastic with Gaussian-smoothed interpolation. Keep as one of the candidates, not the fallback.

After the benchmark, pick the tool with the lowest TRE on the lab's tissue, subject to license / JVM-burden constraints from Open Decisions #4–5.

**API contract — required regardless of which tool wins.**
- `registration.py` MUST return persistent, first-class **transforms** (a dict of `section_id → Transform` object), not just registered images. IHC, mIF, and Xenium morphology images at intermediate z-positions must be warpable using these transforms without re-running registration.
- Transforms are stored as part of the SpatialData container's `transformations` (see Tier 1.5), tied to a shared `microns_3d` coordinate system.
- Registration is computed at one resolution (CODA's 8 µm/px elastic level is the recommended default) and applied at any pyramid level. **Unit-test the transform-rescaling case** — applying a transform computed at 8 µm/px to a level-0 image without rescaling the translation component is a silent failure mode.

```python
# API contract (illustrative, tool-agnostic)
from path3d.registration import register_stack, Transform

transforms: dict[str, Transform] = register_stack(
    manifest_csv="manifest.csv",  # canonical z-coordinates
    backend="valis",              # or "deeperhistreg", "ants", "radon", ...
    target_resolution_um=8.0,
)
# Persisted to SpatialData; reusable for IHC / mIF / Xenium
```

**Pre-registration section QC** (Tier 1, mandatory):
- Detect tissue folds (anomalous stain intensity) — mask during registration
- Detect tears (mask discontinuities)
- Flag missing / damaged sections (small tissue area)
- Accept the manifest CSV as the single source of truth for z-coordinates; never infer z from filename ordering

#### 1.3 Semantic Segmentation (`segmentation.py`)

**Replaces**: `train_image_segmentation.m`, `train_image_segmentation_lung.m`

**Primary path: pathology foundation model encoder + lightweight decoder.** This is the single largest quality lift available and directly addresses the project's biggest non-code risk (annotation budget). 2025 benchmarks consistently show pathology FMs (UNI, UNI2, Virchow2, H-optimus-0) outperform ImageNet ViTs and ImageNet ResNets on segmentation, often reaching equivalent Dice with 5–10× less labeled data.

- **Encoder (default)**: UNI2 (`MahmoodLab/UNI2-h`, ViT-H, January 2025, trained on 200M H&E tiles) frozen. Load via `timm.create_model('hf_hub:MahmoodLab/UNI2-h', pretrained=True, num_classes=0)`. Note: UNI2 weights are gated CC-BY-NC; HuggingFace access request is a project-management item, flag early.
- **Encoder (fallback if UNI2 license is blocking)**: H-optimus-0 (Bioptimus, ViT-G, more permissive license). Virchow2 also viable.
- **Decoder**: lightweight FPN or Mask2Former, trained on lab annotations. `segmentation_models_pytorch` accepts custom encoders; alternatively a 50-line custom FPN suffices.
- **Fine-tuning strategy**: frozen encoder by default. LoRA adapters on the encoder if frozen features prove insufficient — middle ground between frozen and full finetuning.

**Baseline path: DeepLab v3+ from ImageNet (`segmentation_models_pytorch`).**
- Implementation: `smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", classes=num_classes)`
- Used as a sanity-check baseline; the student should know how to do this properly. Document in the same module, behind a `backbone='deeplab_resnet50'` flag.
- Production runs use the FM path unless GPU/annotation budget rules it out (see Open Decisions #1–2).

**Training data:**
- ~50–100 annotated 1024×1024 patches per compartment (FM path) — substantially less than CODA's 6,480 tiles per organ.
- Annotations exported from QuPath as GeoJSON; XML supported only for legacy CODA inputs.
- Augmentation via `albumentations`: rotation, scale (0.8–1.2×), elastic, hue (0.8–1.2× per RGB channel). Stain augmentation only if stain normalization is OFF (see Tier 1.1 benchmark).
- Standard PyTorch loop: `CrossEntropyLoss` (or `DiceLoss` + CE for class imbalance), `AdamW`, validation patience of 5.
- Class balancing via weighted sampling, NOT CODA's overlay-tile recipe — that recipe was designed for a workflow without modern data loaders and adds no value here.

**Inference on WSIs:**
- 1024×1024 tiles with 128–256 px overlap; retain only the inner region per tile (discard border predictions).
- **Tissue-mask gating** (added since plan_review.md): pass the tissue mask to inference and skip tiles that are entirely background. Set background pixels to a dedicated "background" class without running the network on them. Avoids wasted compute and prevents nonsense edge labels from contaminating connected-component analysis downstream.
- Multi-worker `DataLoader` for I/O prefetching; `openslide` reads tiles on-the-fly.
- Stitch predictions into the full label map; write to the SpatialData container as a `(z, y, x)` integer-labels variable at the segmentation resolution (2 µm/px).
- GPU inference is not optional for practical use — see Open Decisions #1.

```python
# API sketch — FM path (default)
import timm
import segmentation_models_pytorch as smp

encoder = timm.create_model('hf_hub:MahmoodLab/UNI2-h', pretrained=True, num_classes=0)
model = build_fm_segmentation(encoder, num_classes=num_classes, decoder='fpn')

# API sketch — baseline path
model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", classes=num_classes)
```

#### 1.4 Cell Detection (`nuclear_detection.py`)

**Replaces**: `cell_detection.m`, `get_nuclear_detection_parameters.m`. (Color deconvolution moved to `utils.py` — see "Architecture" below.)

**Primary: Cellpose-SAM instance segmentation on H&E directly.** Tier 1 produces instance masks (not centroids). The Tier 2 "Cellpose upgrade" path described in earlier drafts collapses into Tier 1 — there is no longer a reason to build a centroid-only path first and replace it later.

- Use `cellpose>=4` (Cellpose-SAM, Stringer & Pachitariu 2025). Pretrained `cpsam` model handles 3-channel H&E without retraining.
- One-line API: `model = cellpose.models.CellposeModel(model_type='cpsam'); masks, _, _ = model.eval(rgb_tile, channels=[0,0])`
- Output: per-tile instance label map; aggregated to a per-section instance mask, with per-instance feature table (centroid, area, Feret diameter, eccentricity, solidity, orientation).
- License: Cellpose 4 is non-commercial — flag this if there's any commercialization angle (Open Decisions #4). Otherwise a non-issue for academic use.
- GPU dependence (~5–10× slower than `trackpy` per WSI; fine for ~200 sections per pilot). See Open Decisions #1.

**Fallback: `trackpy` Crocker-Grier**, kept as a sanity-check benchmark, NOT the default.
- Use case: head-to-head detection-rate comparison against Cellpose-SAM on a held-out section, or environments without a GPU.
- API: `trackpy.locate(-hematoxylin_channel, diameter=11, minmass=100)` after color deconvolution via `path3d.utils.deconvolve(rgb, stain_matrix='HED')`. Note the hematoxylin channel has inverted intensity (dark nuclei = low values).

**Stereological correction (improved over CODA):**
- CODA uses a per-tissue-type constant `D_subtype` (mean nuclear diameter for the subtype). With instance masks, we have the *measured* Feret diameter per nucleus. Use the per-instance form:
  ```
  N_3D = sum_i ( T / (T + D_i) )
  ```
  where `D_i` is the measured Feret diameter of nucleus i's mask and `T` is section thickness (typically 4 µm).
- This is strictly more accurate than CODA's per-subtype constant — `D` varies meaningfully within a compartment.
- Document as `path3d.nuclear_detection.stereological_correct(masks, section_thickness_um=4.0)`.

**Tile-boundary deduplication for instance masks** (different from centroid-based dedup):
- IoU-based mask matching, not centroid distance.
- Pattern: run inference with overlap (e.g., 256 px on 1024 px tiles); in each tile discard any mask whose centroid is in the outer 128 px frame **except** for tiles on the WSI boundary; for masks whose centroid is in the inner region but whose mask area extends into the outer region, keep the mask and clip it to the inner region only. This guarantees each cell's centroid lives in exactly one tile's inner region — no double-counting, no missed cells.
- Worth ~half a day to get right; ~a week of debugging to discover when it's wrong.

```python
# Primary path
from cellpose import models
model = models.CellposeModel(model_type='cpsam')
masks, _, _ = model.eval(rgb_tile, channels=[0, 0])

# Fallback (sanity-check benchmark only)
import trackpy as tp
from path3d.utils import deconvolve
hematoxylin = deconvolve(rgb_tile, stain_matrix='HED')[:, :, 0]
coords = tp.locate(-hematoxylin, diameter=11, minmass=100)
```

#### 1.5 3D Volume Construction (`volume.py`)

**Replaces**: `build_tissue_volume.m`, `build_cell_volume.m`, `combine_z_projections.m`

**Architecture: multimodal lazy container, not a NumPy stack.** This is the most consequential Tier 1 design decision for downstream IHC / mIF / Xenium integration. A single `(Z, Y, X)` label volume forces every additional modality to be its own out-of-band 3D array with its own coordinate management — and IHC sections physically interleave between H&E sections, so their z-coordinates won't align with a uniform stack. Doing this right in Tier 1 is ~1.5 weeks; doing it wrong costs 4–6 weeks of refactoring when Tier 2 starts.

**Default: `SpatialData`-backed multimodal container.**
- Each section is a coordinate on a shared `microns_3d` axis — regardless of whether it carries H&E, IHC, or mIF data.
- Modalities are labeled variables: `H&E_thumbnail`, `tissue_labels`, `nuclear_masks`, `IHC_DAB` (Tier 2), `mIF_channel_*` (Tier 2), etc. Missing data is missing, not zero.
- Adding a new modality is adding a variable on the existing z-axis, not a new pipeline.
- Per-section transformations from registration (Tier 1.2) are stored as part of the SpatialData object's `transformations`, tied to `microns_3d`.
- Storage: lazy zarr store; chunks `(1, 512, 512)` for typical access (one section in z, 512×512 in plane). Label volumes `uint8` (up to 255 classes). Cell-density volumes `float32`. Probability volumes (if stored) `float16`.

**Caveat — verify SpatialData 3D support before committing.** `spatialdata` was designed for 2D + multi-section, not native 3D. Storing a 3D label volume as a `(z, y, x)` array element with a `microns_3d` coordinate system works, but querying along arbitrary 3D planes via the public API may be awkward. The student should test current `spatialdata` 3D capabilities (as of Tier 1.5 implementation) before committing — if insufficient, fall back to a custom `xarray.Dataset` wrapper. Either way the **interface contract** is fixed: multimodal, lazy, z-coordinate-indexed, registration transforms as first-class objects.

**Key behaviors:**
- Stack registered 2D label maps into the SpatialData container along the `z` axis at the segmentation resolution (2 µm/px).
- Resampling: use **nearest-neighbor interpolation** (`order=0`) when resampling label maps. Bilinear/bicubic creates invalid intermediate label values.
- Final 3D volume: 12 µm isotropic voxels (downsample in-plane to match z-spacing) — written as a separate variable.
- Cell coordinates: aggregate (x, y, section_index) into a per-section table, write as `obs` of an `AnnData` with `obsm['spatial_3d']` = (z, y, x) in microns. Optionally bin into a 3D density volume variable.
- Accept the **manifest CSV** as the single source of truth for z-coordinates: columns `[slide_id, file_path, z_microns, was_stained, modality]`. Never infer z from filename ordering.
- Apply gentle z-axis Gaussian smoothing to **registration transforms** (not label maps) to reduce positional oscillation from residual registration error.

**z-interpolation hook** — design the API now even though Tier 1 only implements the simple cases:
- `volume.fill_missing_z(method='nearest'|'linear'|'optical_flow')`
- Tier 1 implements `nearest` and `linear`. Tier 3 adds `optical_flow` (InterpolAI-style). Designing the API to take this argument means downstream code never has to know which method ran.

```python
# API contract (illustrative)
import spatialdata as sd
from path3d.volume import build_volume, fill_missing_z

sdata: sd.SpatialData = build_volume(
    manifest_csv="manifest.csv",
    transforms=transforms,                       # from Tier 1.2
    label_maps=label_maps,                       # from Tier 1.3
    instance_masks=masks,                        # from Tier 1.4
    target_voxel_um=(12.0, 12.0, 12.0),
)
sdata = fill_missing_z(sdata, method='linear')   # Tier 1 default
```

### Tier 1.5: Quantification Layer (New)

The core pipeline produces intermediate outputs (label volumes, nuclear coordinates). This tier converts them into biological measurements — without this, the tool produces images, not science.

#### 1.6 Quantification (`quantification.py`)

**Output format: `AnnData` natively, CSV as a secondary export.** Quantification operates on the SpatialData container from Tier 1.5 and emits two `AnnData` objects:

- **Per-cell**: `obs` = morphometric and microenvironment features (area, eccentricity, Feret, neighborhood composition); `obsm['spatial_3d']` = (z, y, x) in microns; `uns['volumes']` = per-compartment volumetrics
- **Per-lesion**: `obs` = volume, surface area, centroid, bounding box, local cell density, cell-type ratios; `obsm['spatial_3d']` = lesion centroids

This is the lab's downstream stack — labmates can `sc.pl.umap(adata)` on lesion morphometrics the day the pipeline finishes, and the same object joins to Xenium-derived `AnnData` via centroid spatial alignment. Functions like radial profiling, surface-tangential decorrelation, and hotspot/coldspot detection take an AnnData / SpatialData and return augmented versions of the same — they are not standalone CSV writers.

Per-compartment volumetrics:
- Absolute and fractional volumes of each tissue type
- Connected component analysis: number and size distribution of distinct 3D structures (`skimage.measure.label` + `regionprops_table`)
- Surface area of each compartment (`skimage.measure.marching_cubes`)

**Per-lesion feature tables** (as demonstrated in Crawford et al. for 1,476 PanINs):
- For each connected component of a given tissue type: volume, surface area, centroid, bounding box
- Local cell density per cell type within/around each lesion
- Cell type ratios (e.g., FOXP3+/CD45+, FOXP3+/CD3+)
- Local tissue composition in configurable radius around each lesion
- Size stratification (e.g., small vs large lesions) for comparative analysis
- Export as a DataFrame where each row is a lesion

Cell density statistics:
- Per-tissue-type cell density (cells/mm³)
- 3D kernel density estimation at configurable bandwidths
- Density gradients (where cellularity changes most rapidly)
- **Local sphere density**: tissue composition and cell density within configurable radius (typically 150 µm) around any point or structure, using `scipy.ndimage.distance_transform_edt`

**Radial (concentric ring) microenvironment profiling** (as in Crawford et al. and Forjaz et al.):
- For any labeled structure (tumor, lesion, duct, etc.), dilate outward in configurable increments (e.g., 10 µm)
- Quantify cellular/tissue composition at each distance interval out to a maximum radius (e.g., 500 µm)
- Compare proximal vs distal microenvironment composition
- 3D radial analysis using `scipy.ndimage.distance_transform_edt` — Crawford et al. demonstrated that 2D radial measurements dramatically over/underestimate the true 3D immune density

**Surface-tangential decorrelation analysis** (as in Crawford et al.):
- For structures with complex 3D surfaces (e.g., PanINs), measure how a metric (e.g., immune density) changes as you traverse the surface
- Compute decorrelation length: distance along the surface required for the metric to change by 25%, 50%, or 100%
- Answers "how spatially heterogeneous is the microenvironment around this structure?"

**Hotspot/coldspot detection** (as in Crawford et al.):
- Algorithmically identify spatial extremes of any quantitative metric around structures of interest
- Enforce minimum separation (e.g., 0.5 mm) between selected points to avoid clustering
- Export high-resolution image patches (e.g., 0.5×0.5 mm) at each location for pathologist review
- Useful for directing expert attention to the most interesting regions

**2D-vs-3D comparison utility**:
- Given a full 3D reconstruction, sample random 2D cross-sections and compare against the true 3D measurement
- Demonstrates the value of 3D analysis; useful for publications and for evaluating whether 2D shortcuts are acceptable for a given application

**Per-cell local-neighborhood feature vector** (added since plan_review.md — direct input to dormancy / niche discovery analysis):
- For each cell, compute within radii R = {50, 100, 200} µm: composition fractions per compartment, total cell density, nearest-neighbor distance per cell type, Shannon entropy of tissue-type composition.
- Stored as columns of the per-cell AnnData `obs`. This is the input to downstream niche-discovery clustering (Leiden, latent-space methods) — spec it now or it becomes a one-off script later.

All outputs additionally exported as **pandas DataFrames / CSV** for non-Python users; AnnData is canonical.

#### 1.7 Section QC (`preprocessing.py`)

Automated quality control before registration:
- Report tissue area per section, flag outliers (damaged/missing sections)
- Detect tissue folds (anomalously high stain intensity regions)
- Detect tears (discontinuities in tissue mask)
- After registration: report residual error per section pair
- After segmentation: report class proportions per section, flag sudden changes
- Generate a QC summary report

### Tier 2: Quality, Integration, and Modern Methods

#### 2.1 Multi-Stain IHC and mIF Co-Registration and Quantification

**Based on validated workflows from Crawford et al. (2024) and Forjaz et al. (2025), generalized to mIF**

The Tier 1 SpatialData container, transform persistence, and `utils.deconvolve()` already provide the architecture; Tier 2 fills in the IHC/mIF-specific operations:

1. **H&E first, IHC/mIF second**: Register all H&E sections via the Tier 1 pipeline. Align each IHC or mIF section to the registered H&E stack using the same backend (registration is pluggable per Tier 1.2). Crawford et al. tested 3 approaches and confirmed H&E-first is optimal.
2. **Add the IHC/mIF section as a new variable on the existing z-axis** of the SpatialData container — no rework of Tier 1 outputs.
3. **Color deconvolution** for IHC: reuse `path3d.utils.deconvolve(rgb, stain_matrix='HDAB')` to isolate antibody channels.
4. **Cell-level positivity**: project nuclear instance masks (from Tier 1.4 Cellpose-SAM, warped through registration transforms) onto the IHC/mIF channel; assign per-cell positivity. K-medoids clustering on the antibody-channel intensity per cell (as in Crawford et al. for CD45, CD3/FOXP3) avoids manual thresholding. For mIF, per-channel intensity is already quantitative — direct thresholding or Gaussian-mixture is appropriate.
5. **Optional: train a binary IHC positivity model** (as in Forjaz et al.'s third cascade model) when intensity-based methods are insufficient.

Supported staining schedules:
- Every Nth section IHC (e.g., 1 in 8 for p53, 1 in 8 for Ki67 — as in Forjaz et al.)
- Alternating H&E and IHC sections
- Multiple IHC markers on separate intervening sections
- mIF panels (one section, multiple channels) — each channel becomes a variable on the SpatialData container at that section's z-coordinate

#### 2.2 Hierarchical / Cascaded Segmentation Models

**Based on Forjaz et al. (2025) three-model cascade architecture**

Design the segmentation module to support composable model pipelines:
1. **Coarse tissue model**: segment major tissue compartments (epithelium, stroma, vessels, fat, nerve, etc.)
2. **Fine subtype models**: operate within specific compartments (e.g., secretory vs ciliated epithelium within the epithelial compartment; normal duct vs PanIN within ducts)
3. **Marker signal models**: detect IHC positivity (p53+, Ki67+) on co-registered IHC sections

This is a general design pattern applicable to any organ. Each model in the cascade receives a mask from the previous level.

#### 2.3 Nuclear Instance Segmentation — moved to Tier 1

Cellpose-SAM is now Tier 1.4's primary path. This Tier 2 item is retained only as a placeholder for alternatives the student may benchmark against (HoVer-Net, StarDist with H&E fine-tuning, nnU-Net) if Cellpose-SAM is inadequate on a specific tissue. StarDist still requires TensorFlow (dependency conflict risk) — avoid unless its star-convex prior is materially better on the lab's tissue.

#### 2.4 H&E-to-IHC Transfer Models

**Based on Crawford et al. (2024) power-law immune density estimation**

When IHC is available on only a subset of samples, learn a mapping function from H&E-derived features (e.g., stromal cell density) to IHC-derived measurements (e.g., CD45+ cell density):
- Calibrate on paired H&E + IHC samples using cross-validated regression (linear, exponential, power-law)
- Apply the best-fit model to predict molecular features across the entire H&E-only cohort
- This massively reduces IHC burden for large studies

#### 2.5 Pathology Foundation Model Encoders — moved to Tier 1

Now Tier 1.3's primary path (UNI2 / H-optimus-0). Retained here only as a pointer to alternatives if the chosen FM is inadequate or licensing changes — Phikon, CONCH, Virchow2 are the obvious next options.

#### 2.6 Xenium / snPATHO-seq / Spatial Transcriptomics Integration

**Lab-native priority — Xenium is the lab's primary spatial platform, not Visium.**

The Tier 1 SpatialData container makes this an addition rather than a port:

- **Register Xenium morphology images onto the H&E scaffold** using the persisted Tier 1.2 transforms. Xenium's morphology DAPI/HE channels can be aligned to the nearest H&E section's registered space, then projected to `microns_3d`.
- **Per-Xenium-cell tissue label** by indexing the Tier 1.3 label volume at each cell's transformed (z, y, x). Add as `obs['pycoda_compartment']` on the Xenium AnnData.
- **Per-Xenium-cell microenvironment vector** by reusing the Tier 1.6 neighborhood-feature function. Lets the lab cluster Xenium cells by their 3D-aware niche, not just their 2D-section neighborhood.
- **Tissue-type-aware Xenium analysis** in `squidpy` / `scanpy` using the joined labels.
- **3D-mapped lesion guidance for sectioning** (Forjaz et al. used this for Visium CytAssist placement; same logic applies to Xenium section selection from a CODA-reconstructed block).
- **inferCNV-style CNV inference** from Xenium / snPATHO-seq is upstream of path3d but the joined AnnData makes downstream visualization on the 3D volume trivial.

snPATHO-seq integration is structurally similar — the snPATHO-seq cells come from a specific section that lands at a known `z_microns`, so each cell gets a (z, y, x) coordinate via the section's registration transform.

Visium / Visium CytAssist support falls out of the same code path if needed for collaborator data; it is not a lab priority.

#### 2.7 AnnData / SpatialData Export — moved to Tier 1

Now Tier 1.5 / 1.6's canonical output format. Retained as a Tier 2 entry only because additional `napari-spatialdata` integration (interactive 3D ROI analysis, brushed selections) is a non-trivial extension worth keeping in scope.

#### 2.8 Virtual Re-Sectioning Along Tubular Axes

**Based on Forjaz et al. (2025) lumen skeletonization**

For tubular structures (ducts, vessels, fallopian tube, bronchi, nephrons):
1. Skeletonize the lumen to extract the center path (`skimage.morphology.skeletonize_3d`)
2. Fit a smooth spline to the skeleton
3. Generate virtual cross-sections **perpendicular to the path** at configurable intervals
4. Quantify cell/tissue composition per virtual section as a function of distance along the structure

Forjaz et al. generated up to 10,255 virtual sections per specimen. This enables analysis in biologically meaningful orientations that are impossible to achieve physically.

#### 2.9 Digital Subsampling Simulation

**Based on Forjaz et al. (2025) virtual SEE-FIM protocol simulation**

Given a complete 3D reconstruction, simulate what you'd see if you only sampled N sections at various intervals and orientations:
- Vary sampling density from 1 to full coverage
- For each density, count how many known features (lesions, structures) are detected
- Compute false-negative rate as a function of sampling density
- Answers: "how many sections do I need to section to reliably detect features of size X?"

Applications: optimize clinical sampling protocols, power analysis for study design, evaluate existing sampling adequacy.

#### 2.10 Pre-trained Model Zoo

Retrain and distribute PyTorch segmentation models for validated tissues (pancreas, lung, skin, liver, fallopian tube) using CODA's training recipe. This lowers the barrier to entry from "annotate for days" to "run the pipeline immediately."

#### 2.11 Parameter Calibration and Validation

- Cell detection calibration: user provides manually counted regions, optimize parameters via grid search
- Registration validation: target registration error (TRE) using fiducial landmarks (Crawford et al. used 100 landmarks on 50 image pairs)
- Validation against MATLAB outputs: compare per-stage (registration overlap, Dice per class, cell count/coordinate agreement)
- Document quantitative parity metrics

### Tier 3: Extensions

#### 3.1 Modern Segmentation Architectures
- **SegFormer** (via HuggingFace `transformers`): transformer-based, better long-range context
- **SAM** (Segment Anything): interactive annotation via point/box prompts

#### 3.2 InterpolAI-style Z-Interpolation
- Deep learning optical flow interpolation between sections to improve z-resolution
- Now confirmed as a **production component** of the Kiemen lab pipeline (used in Forjaz et al. 2025 for missing section restoration)
- Slots between registration (Stage 2) and volume construction (Stage 4)
- Design module interfaces now to accommodate this without refactoring

#### 3.3 Interactive Visualization with napari
- Interactive 3D browsing of tissue volumes
- Layer tissue labels, cell coordinates, and raw images
- ROI-based analysis: draw a region, compute statistics for that region

```python
import napari
viewer = napari.Viewer()
viewer.add_labels(tissue_volume, name="tissue types")
viewer.add_points(cell_coords, name="nuclei", size=2)
```

#### 3.4 Network Topology Extraction
- Skeletonize tubular structures (ducts, vessels) with `skimage.morphology.skeletonize_3d`
- Build graph representation (nodes = branch points, edges = tubular segments)
- Compute topology: branching angles, total length, connectivity

#### 3.5 Lesion Growth Modeling
**Based on Forjaz et al. (2025) and Kiemen et al. (2022)**
- Fit power-law growth models to lesion volume distributions
- Compare growth dynamics across tissue types and conditions (e.g., PanIN polyclonal merging vs STIC monoclonal growth)
- Kolmogorov-Smirnov test for goodness of fit

#### 3.6 Multiplexed Proteomics Integration (CODEX/IMC)
**Based on Forjaz et al. (2025) and Crawford et al. (2024)**
- 25-plex CODEX or 38-plex IMC on selected sections
- DAPI channel for nuclear segmentation, antibody intensity per cell
- Unsupervised clustering (UMAP + Leiden/k-medoids) for cell phenotyping
- PAGA graph abstraction for cell phenotype connectivity analysis
- Spatial mapping of phenotype distributions relative to 3D structures

#### 3.7 Batch Processing and Reporting
- Config-file-driven batch processing across specimens
- Standardized output directory structure with provenance tracking
- Automated report generation (QC summary, tissue composition, morphometrics)

#### 3.8 GPU Acceleration and Scalability
- `cucim` (RAPIDS): GPU-accelerated WSI reading (Linux only)
- `cupy`: GPU-accelerated numpy for volume operations
- `torch.compile` or TensorRT for segmentation inference optimization

## MATLAB → Python Translation Cheat Sheet

| MATLAB | Python | Notes |
|--------|--------|-------|
| `imread` / `imwrite` | `tifffile.imread` / `skimage.io.imsave` | For standard images |
| `openslide_read_region` | `openslide.OpenSlide.read_region()` | Returns RGBA — always strip alpha channel |
| `imresize` | `skimage.transform.resize` | Specify `anti_aliasing=True`; use `order=0` for label maps |
| `rgb2gray` | `skimage.color.rgb2gray` | Returns float [0,1] not uint8 |
| `imbinarize` / `graythresh` | `skimage.filters.threshold_otsu` | Otsu thresholding |
| `bwareaopen` | `skimage.morphology.remove_small_objects` | |
| `imfill` | `scipy.ndimage.binary_fill_holes` | |
| `radon` | `skimage.transform.radon` | For CODA's rotation-finding step |
| `normxcorr2` | `scipy.signal.fftconvolve` or `cv2.matchTemplate` | For full correlation map; `phase_cross_correlation` only returns peak shift |
| `imwarp` | `skimage.transform.warp` or `scipy.ndimage.map_coordinates` | |
| `pkfndW` (particle tracking) | `cellpose.models.CellposeModel('cpsam').eval()` | path3d primary; gives instance masks. `trackpy.locate()` kept as fallback |
| `trainNetwork` (DeepLab) | PyTorch training loop on FM encoder + decoder | UNI2 / H-optimus-0 frozen + lightweight FPN decoder; or `smp.DeepLabV3Plus` baseline |
| `semanticseg` | `model(input_tensor)` | Standard PyTorch inference; gate by tissue mask |
| `isosurface` | `skimage.measure.marching_cubes` | Returns verts/faces |
| `patch` (3D surface) | `pyvista.PolyData` + `.plot()` | Or `vedo` |
| `bwlabeln` | `skimage.measure.label` | 3D connected component labeling |
| `cat(3, ...)` / 3D array | `numpy.stack(arrays, axis=0)` | |
| `A(:)` (columnwise flatten) | `A.ravel(order='F')` | MATLAB is column-major (Fortran order); NumPy default is row-major (C order) |

## Known Challenges and Gotchas

### Registration: Algorithmic Divergence
VALIS uses fundamentally different algorithms than CODA (feature-based vs Radon transform, B-spline vs tiled rigid elastic). CODA's multi-reference strategy is a key innovation for preventing z-drift. Validate VALIS outputs against MATLAB outputs early and thoroughly — overlay alternating sections, check for tissue drift, compute landmark distances.

### Registration: Failure Modes
- **Tissue folds**: double-thickness staining confuses both cross-correlation and feature matching. Detect folds (anomalous stain intensity) and mask them during registration.
- **Tissue tears**: create discontinuities that elastic registration may try to "close," distorting surrounding tissue. Mask torn regions.
- **Missing sections**: create z-gaps. Accept a section manifest with z-coordinates; interpolate or leave gaps.
- **z-drift**: sequential pairwise registration compounds small errors. Use reference-based alignment (VALIS does this) or periodically re-anchor to a reference section.

### WSI Memory Management
Whole slide images are multi-gigabyte (2-5 GB compressed, 20-50 GB uncompressed). Never load an entire WSI into RAM. Always use tiled reading (`openslide.read_region()`) and process tile-by-tile. Use a multi-worker `DataLoader` for I/O prefetching during segmentation inference.

### openslide Returns RGBA
`openslide.OpenSlide.read_region()` returns a PIL Image in **RGBA mode** (4 channels), not RGB. Every call must be followed by `.convert("RGB")` or `np.array(tile)[:, :, :3]`.

### Image Dtype Conventions
- `scikit-image` functions expect float64 images in [0, 1]. Use `skimage.img_as_float` for conversion.
- `cv2` functions expect uint8 images in [0, 255].
- **Pick one library and stick with it.** Recommend scikit-image throughout to avoid confusion.
- `cv2` reads images in **BGR** order; every other library uses RGB.

### Stain Normalization Sensitivity
Stain normalization can help, hurt, or do nothing depending on the segmentation backbone. Foundation-model encoders (UNI2, Virchow2) trained on multi-institution data are partially robust to stain variance and 2025 benchmarks have shown normalization can *reduce* their embedding consistency in some cases. ImageNet-pretrained networks usually benefit from normalization. **Run the benchmark**: segmentation with and without normalization on a held-out section under the chosen backbone; pick the default by Dice. Don't pick by assumption.

### Pretrained Weight Transfer
MATLAB CODA DeepLab weights do not transfer to PyTorch (different serialization, different layer naming). Don't try. The path3d primary path uses a frozen pathology FM encoder (UNI2 / H-optimus-0) plus a decoder trained on the lab's own annotations — not a port of CODA's weights and not from-scratch training of a new ImageNet network. The DeepLab v3+ baseline path *does* train from ImageNet-pretrained weights for sanity-check comparison.

### Coordinate Systems
Define a project-wide convention and enforce it:
- All coordinates are **0-based** (MATLAB uses 1-based)
- 2D: **(row, col) = (y, x)** following NumPy/scikit-image convention
- 3D: **(z, row, col) = (section_index, y, x)**
- Cell coordinates: arrays of shape (N, 3) with columns [z, y, x]
- Physical units: **microns**, unless otherwise stated
- MATLAB arrays are column-major (Fortran order); NumPy is row-major (C order). Use `ravel(order='F')` to match MATLAB flattening behavior.

### Label Map Interpolation
When downsampling label maps (categorical data), **always use nearest-neighbor interpolation** (`order=0` in scikit-image, `cv2.INTER_NEAREST` in OpenCV). Bilinear/bicubic interpolation creates invalid intermediate label values (e.g., averaging "acini"=3 and "collagen"=7 gives a meaningless 5).

### Tile Boundary Artifacts
When processing WSIs tile-by-tile:
- **Segmentation**: use 128–256 pixel overlap for 1024×1024 tiles. Retain only the inner prediction region; discard border predictions.
- **Cell detection (Cellpose-SAM instance masks)**: use IoU-based mask matching for boundary deduplication, not centroid distance. Pattern documented in Tier 1.4.
- **Cell detection (`trackpy` fallback only)**: use overlap ≥ 2× `min_distance`; deduplicate via spatial non-maximum suppression.
- **Context loss**: tile-based segmentation loses tissue-level context. A small duct may be classified differently depending on surrounding tissue. Consider larger tiles or multi-scale inference for difficult cases.

### Tissue Mask Must Gate Segmentation Inference
Don't run the segmentation network on background pixels. Pass the tissue mask to inference and skip tiles that are entirely background; assign background a dedicated "background" class. Otherwise nonsense edge labels contaminate connected-component analysis downstream and waste compute on hundreds of empty tiles per WSI.

### Registration Transform Resolution Rescaling
Registration is computed at one resolution (8 µm/px elastic level by default) and applied at any pyramid level. Applying a transform computed at 8 µm/px to a level-0 image without rescaling the translation component is a silent failure mode — outputs look "almost right" but are systematically offset by 4× (or whatever the level ratio is). **Add a unit test** that constructs a known transform at one level, applies it at another, and checks the warped image against ground truth. This is one of the most common silent bugs in multi-resolution histology pipelines.

### Manifest CSV is the Single Source of z-Truth
Sections are not always cut in physical order in the histology core. Two sections per slide is common; broken slides get re-cut. The student must accept a `manifest.csv` with columns `[slide_id, file_path, z_microns, was_stained, modality]` and never infer z from filename ordering. This is mentioned in Tier 1.5 but bears repeating as a gotcha — pipelines that infer z from filenames will silently fail on real lab inputs.

### openslide File Descriptor Exhaustion
`openslide.OpenSlide(path)` opens a file handle. Opening 200 of them in a list comprehension to extract metadata will exhaust file descriptors on Linux and crash on macOS. Use a context manager pattern, or extract metadata once and cache to the manifest CSV.

### VALIS / Bio-Formats JVM Lifecycle
If VALIS is the chosen registration backend, always wrap it in `try/finally` with `registration.kill_jvm()` — otherwise pipelines hang on cluster nodes after completion and the lab's HPC scheduler will eventually start emailing the PI.

### dtype and zarr Chunking Conventions
- Label volumes: `uint8` (up to 255 classes — sufficient for any realistic compartment count)
- Cell-density volumes: `float32`
- Probability volumes (if stored): `float16` — fine for probabilities, lossless to ~10⁻³, halves storage
- zarr chunks: `(1, 512, 512)` for typical access patterns (one section in z, 512×512 in plane)

Getting these wrong silently halves performance or doubles storage cost. Make these defaults in the SpatialData write path, don't push the choice onto users.

### `spatialdata` 3D Support Caveat
`spatialdata` was designed for 2D + multi-section, not native 3D. Storing a 3D label volume with a `microns_3d` coordinate system works, but querying along arbitrary 3D planes via the public API may be awkward as of writing. The student should test current 3D capabilities before committing in Tier 1.5; if insufficient, fall back to a custom `xarray.Dataset` wrapper. The interface contract is fixed regardless.

### Compiled MATLAB Files
CODA's `pkfndW.p` and `im2mat.p` are compiled MATLAB p-code that cannot be inspected. `trackpy` is the closest Python equivalent (Crocker-Grier). However: path3d Tier 1 defaults to Cellpose-SAM, not Crocker-Grier — `trackpy` is kept only as a sanity-check fallback. Don't burn time validating against `pkfndW`; validate against the lab's own manual counts on real tissue.

### VALIS Installation (if VALIS wins the registration benchmark)
`pip install valis-wsi` requires system dependencies:
- **libvips**: `brew install libvips` (macOS) or `apt install libvips42` (Ubuntu)
- **Java Runtime Environment**: `brew install openjdk` (macOS) or `apt install default-jre` (Ubuntu) — required for Bio-Formats slide reader
- These are non-trivial to install. Use conda (`conda install -c conda-forge openslide-python libvips openjdk`) to simplify.

If a non-JVM tool (DeeperHistReg, ANTs/SimpleElastix) reaches comparable TRE on the lab's pilot, prefer it — the JVM lifecycle and license (GPL-3) burden is real. See Open Decisions #4–5.
