# CODA: Overview of the Original Implementation

## Publication Context

**CODA** was published by Kiemen, Braxton, Grahn, et al. in *Nature Methods* (2022), from the labs of Ashley Kiemen and Denis Wirtz at Johns Hopkins University. It is part of the HuBMAP (Human BioMolecular Atlas Program) consortium.

CODA enables quantitative 3D reconstruction of large tissue specimens (up to multi-cm³) at subcellular resolution from standard histological preparations — serially sectioned, H&E-stained tissue. It has been validated on pancreas, skin, lung, and liver tissues, identifying structures such as ductal epithelium, pancreatic cancer precursors (PanIN), PDAC, smooth muscle, acini, fat, collagen, islets of Langerhans, and lymph nodes.

## Pipeline Architecture

```
Whole Slide Images (H&E, .svs/.ndpi)
        │
        ▼
┌─────────────────────┐
│  1. PREPROCESSING    │  Downsampling, tissue mask generation,
│                      │  H&E stain normalization
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. REGISTRATION     │  Global rigid (Radon transform + cross-correlation)
│                      │  + Local elastic registration (tiled)
└────────┬────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│              PARALLEL PATHS                 │
│                                             │
│  3a. SEMANTIC SEGMENTATION                  │
│      DeepLab v3+ / ResNet50                 │
│      → Tissue type labels per pixel         │
│                                             │
│  3b. CELL DETECTION                         │
│      Color deconvolution + particle tracking│
│      → Nuclear coordinates (x, y, z)        │
└────────────────┬───────────────────────────┘
                 │
                 ▼
┌─────────────────────┐
│  4. 3D VOLUME        │  Consolidate into 3D matrices
│     CONSTRUCTION     │  Voxel size: 12 × 12 × 12 µm
│     & VISUALIZATION  │  MATLAB isosurface/patch rendering
└─────────────────────┘
```

## Stage 1: Preprocessing

### Resolution Levels

Different pipeline stages operate at different resolutions:
- **80 µm/pixel**: Global rigid registration
- **8 µm/pixel**: Elastic registration (local tile matching)
- **2 µm/pixel**: Semantic segmentation and cell detection
- **12 µm isotropic**: Final 3D volume

### Downsampling
Whole slide images (typically .svs or .ndpi format at 20x–40x magnification) are read using **Openslide** and downsampled to the appropriate resolution for each pipeline stage.

**MATLAB script**: `create_downsampled_tif_images.m`

### Tissue Mask Generation
Binary masks are generated to distinguish tissue from background (glass slide). CODA uses a **dual-criterion approach**: regions with **low green channel intensity** AND **high RGB standard deviation** are classified as tissue. This is more robust than simple grayscale thresholding for H&E-stained sections.

**MATLAB script**: `calculate_tissue_ws.m`

### Microns-Per-Pixel Calibration
The physical pixel size (microns per pixel) is extracted from slide metadata to ensure correct spatial scaling throughout the pipeline.

**MATLAB script**: `get_mpp_of_image.m`

### H&E Stain Normalization
H&E staining varies between slides due to preparation differences. Stain normalization reduces this heterogeneity before both registration and segmentation.

CODA uses a custom approach: **k-means clustering with 100 clusters** on optical densities to identify stain vectors. The most blue-favored cluster is identified as hematoxylin, the most red-favored as eosin. Background optical density is computed as the inverse of the average of H and E optical densities. This is distinct from standard methods like Macenko (SVD-based) or Vahadane (sparse NMF).

**MATLAB script**: `deconvolve_histological_images.m` (also used in cell detection)

## Stage 2: Image Registration

Registration aligns serial sections into a coherent 3D stack. This is a two-phase process:

### Phase 1: Global Rigid Registration (at 80 µm/pixel)
CODA decomposes rotation and translation into separate steps:
1. **Rotation finding via Radon transform**: The Radon transform of each image is computed, and cross-correlation of Radon transforms at discrete angles (0–359°) identifies the rotation angle between sections. This is more robust than direct image cross-correlation for large rotations.
2. **Translation finding**: After rotating to the identified angle, cross-correlation of the rotated images determines the (x, y) translation.
3. This yields a global rigid transformation (rotation + translation).

### Multi-Reference Strategy
Rather than registering each section only to its immediate neighbor (which compounds errors through the z-stack), CODA registers each image against **three nearby references**: sections n±(m+1), n±(m+2), and n±(m+3). The registration with the **best pixel-to-pixel correlation** is retained. The **center image** of the stack serves as the registration anchor, and all images are registered into that coordinate system. This is a key innovation that prevents z-drift accumulation and handles tissue defects.

### Phase 2: Local Elastic Registration (at 8 µm/pixel)
After rigid alignment:
1. Each image is divided into tiles at **1.5 mm intervals**
2. Local deformations are computed per tile using cross-correlation
3. Results are **interpolated** across the full image and **smoothed with Gaussian filtering**
4. This corrects for non-rigid tissue deformations (folding, tearing, compression)

**MATLAB scripts**:
- `calculate_image_registration.m` — Computes both rigid and elastic registration transforms
- `apply_image_registration.m` — Applies computed transforms to full-resolution images

**Key base functions** (in `image registration base functions/`):
- `calculate_global_reg.m` — Global rigid registration (Radon + cross-correlation)
- `calculate_elastic_registration.m` — Tiled elastic registration
- `xcorrf2.m` — Fast 2D cross-correlation
- `preprocessing.m` — Image preprocessing for registration
- `register_global_im.m` — Apply global registration to images
- `reg_ims_ELS.m` — Apply elastic registration
- `make_final_grids.m` — Construct interpolated deformation grids

### Registration Output
- Transformation matrices (per section pair)
- Registered (aligned) image stack

## Stage 3a: Semantic Segmentation

### Model Architecture
- **DeepLab v3+** with a **ResNet50** backbone
- Implemented using MATLAB's Deep Learning Toolbox
- Trained per-organ (separate models for pancreas, lung, etc.)

### Training Data Preparation
The training data construction is a specific class-balancing strategy:
1. **Seven tissue images equally spaced** within each sample are **manually annotated** in Aperio ImageScope (XML annotations), with **50 examples of each tissue subtype** per image
2. Blank tiles of **9000 × 9000 × 3 pixels** are created
3. Annotated tissue patch bounding boxes of the **least represented class** are randomly overlaid until the tile is **>65% full** and class pixel counts are **approximately equal**
4. Large tiles are cut into **324 tiles of 500 × 500 × 3 pixels**
5. **20 large images** are built (half with augmentation), producing **6,480 training images**
6. **5 additional images** produce **1,620 validation images**
7. **Augmentation**: rotation, scaling (0.8–1.2×), hue augmentation (0.8–1.2× per RGB channel)
8. Training stops at **validation patience of 5**
9. If >90% precision and recall is not achieved, **additional annotations are collected** and training is repeated
10. H&E stain normalization is applied to training data to reduce inter-slide variability

**MATLAB scripts**:
- `train_image_segmentation.m` — General tissue segmentation model training
- `train_image_segmentation_lung.m` — Lung-specific segmentation training variant

### Segmentation Classes (Pancreas Example)
Nine classes are identified in pancreatic tissue:
1. Normal ductal epithelium
2. Pancreatic cancer precursors (PanIN)
3. Pancreatic ductal adenocarcinoma (PDAC)
4. Smooth muscle
5. Acini
6. Fat (adipose)
7. Collagen
8. Islets of Langerhans
9. Lymph nodes

### Inference
The trained model is applied to each registered section, producing a per-pixel label map. These maps are then used to build the 3D tissue volume.

## Stage 3b: Cell / Nuclear Detection

### Algorithm (at 2 µm/pixel)
Cell detection does **not** use deep learning. Instead, it uses the **Crocker-Grier particle tracking algorithm** (originally developed for colloidal physics):

1. **Color deconvolution** separates the hematoxylin channel from H&E images (hematoxylin stains nuclei dark purple/blue)
2. **Intensity normalization** standardizes the hematoxylin channel
3. **Bandpass filtering** followed by detection of **2D intensity minima** of a designated size and distance in the hematoxylin channel (nuclei appear as dark spots = intensity minima)
4. The core detection is implemented in compiled MATLAB p-code: `pkfndW.p` (peak finding) and `im2mat.p` (image-to-matrix conversion) from the Crocker-Grier particle tracking toolkit
5. Processing speed: approximately **90 seconds per whole slide image**

This approach is deliberately training-free — it works across tissues without needing annotated nuclear data.

**MATLAB scripts**:
- `cell_detection.m` — Core nuclear detection
- `deconvolve_histological_images.m` — Color deconvolution for hematoxylin isolation
- `get_nuclear_detection_parameters.m` — Optimizes detection thresholds using manual annotations
- `manual_cell_count.m` — Records ground truth nuclear counts for validation
- `make_cell_detection_mosaic.m` — Creates composite images for detection parameter calibration

**Key base functions** (in `cell_detection_base_functions/`):
- `pkfndW.p` — Compiled Crocker-Grier peak finding (opaque, cannot be inspected)
- `im2mat.p` — Compiled image-to-matrix conversion (opaque)
- `colordeconv_log10.m` — Log-space color deconvolution
- `calculate_optimal_vals.m` — Optimize detection parameters
- `cell_cell_dist.m` — Inter-cell distance calculations

### Stereological Cell Count Correction
To convert 2D nuclear counts to accurate 3D estimates, CODA applies a stereological correction:

**C_3D = Σ(C_image × T / (T + D_subtype))**

Where T is section thickness (4 µm) and D_subtype is the measured nuclear diameter per tissue subtype (100 nuclei measured per subtype per case). This accounts for the bias that larger nuclei appear in more sections, preventing systematic overcounting of tissue types with larger nuclei.

### Cell Coordinate Registration
Detected nuclear coordinates are transformed using the registration transforms computed in Stage 2, aligning them to the common 3D coordinate system.

**MATLAB script**: `register_cell_coordinates.m`

## Stage 4: 3D Volume Construction and Visualization

### Tissue Volume
Segmented label maps from each registered section are stacked into a 3D matrix at a voxel resolution of **12 × 12 × 12 µm**.

**MATLAB scripts**:
- `build_tissue_volume.m` — Constructs the 3D tissue label matrix
- `combine_z_projections.m` — Merges multiple z-plane projections

### Cell Volume
Registered nuclear coordinates are consolidated into a 3D coordinate matrix.

**MATLAB script**: `build_cell_volume.m`

### Visualization
3D renderings are generated using MATLAB's built-in functions:
- `isosurface` — Extracts surface meshes from the label volume
- `patch` — Renders colored surfaces for each tissue type

Each tissue type is assigned a unique RGB color for visualization.

**Original script**: `plot_3D_tiss.m`

## Complete MATLAB Script Inventory

### Updated Version (December 2023) — 16 scripts
| Script | Pipeline Stage | Purpose |
|--------|---------------|---------|
| `create_downsampled_tif_images.m` | Preprocessing | Convert WSIs to lower magnification TIFFs |
| `calculate_tissue_ws.m` | Preprocessing | Generate tissue/background masks |
| `get_mpp_of_image.m` | Preprocessing | Extract microns-per-pixel from metadata |
| `deconvolve_histological_images.m` | Preprocessing / Cell detection | H&E color deconvolution and normalization |
| `calculate_image_registration.m` | Registration | Compute rigid + elastic transforms |
| `apply_image_registration.m` | Registration | Apply transforms to images |
| `train_image_segmentation.m` | Segmentation | Train DeepLab tissue classifier |
| `train_image_segmentation_lung.m` | Segmentation | Train lung-specific classifier |
| `cell_detection.m` | Cell detection | Detect nuclei via particle tracking |
| `get_nuclear_detection_parameters.m` | Cell detection | Optimize detection parameters |
| `manual_cell_count.m` | Cell detection | Record ground truth counts |
| `make_cell_detection_mosaic.m` | Cell detection | Calibration composite images |
| `register_cell_coordinates.m` | Cell detection | Transform cell coords to registered space |
| `build_tissue_volume.m` | Volume construction | Stack segmentation maps into 3D |
| `build_cell_volume.m` | Volume construction | Stack cell coordinates into 3D |
| `combine_z_projections.m` | Volume construction | Merge z-projections |

### Original Upload — 8 scripts
| Script | Purpose |
|--------|---------|
| `register_images.m` | Image registration (predecessor to calculate/apply split) |
| `save_images_elastic2.m` | Save elastically registered images |
| `normalize_HE.m` | H&E stain normalization |
| `make_training_deeplab.m` | Prepare DeepLab training data |
| `HE_cell_count.m` | Nuclear detection in H&E |
| `register_cell_coordinates.m` | Transform cell coordinates |
| `save_cell_coordinates_registered.m` | Export registered cell coordinates |
| `plot_3D_tiss.m` | 3D tissue visualization |

### Visium Integration (May 2024) — 3 files
| Script | Purpose |
|--------|---------|
| `HE_cell_count_visium.m` | Nuclear detection adapted for Visium H&E |
| `register_CODA_segmentation_to_visium_slide.m` | Register CODA outputs to Visium spatial coordinates |
| `README.txt` | Integration workflow instructions |

### Base Function Libraries (in subdirectories)

The repository contains ~40 additional base functions organized into three subdirectories. These implement the core algorithmic logic called by the top-level scripts.

**Registration base functions** (20 files):
`bpassW.m`, `calculate_elastic_registration.m`, `calculate_global_reg.m`, `calculate_global_reg_IHC.m`, `calculate_transform.m`, `find_tissue_area.m`, `gcnt.m`, `getImLocalWindowInd_rf.m`, `get_ims.m`, `group_of_reg.m`, `im2mat.p`, `invert_D.m`, `make_final_grids.m`, `mskcircle2_rect.m`, `pad_im_both2.m`, `preprocessing.m`, `reg_ims_ELS.m`, `reg_ims_com.m`, `register_global_im.m`, `xcorrf2.m`

**Segmentation base functions** (15 files):
`build_model_tiles.m`, `calculate_tissue_space.m`, `combine_tiles_density_shuffle.m`, `deeplab_classification.m`, `fill_annotations_file.m`, `load_xml_file.m`, `load_xml_loop.m`, `make_check_annotation_classified_image.m`, `make_cmap_legend.m`, `make_confusion_matrix.m`, `random_augmentation.m`, `save_annotation_bounding_boxes.m`, `test_model_performance.m`, `train_deeplab.m`, `xml2struct2.m`

**Cell detection base functions** (5 files):
`calculate_optimal_vals.m`, `cell_cell_dist.m`, `colordeconv_log10.m`, `im2mat.p`, `pkfndW.p`

Note: `.p` files are compiled MATLAB p-code and cannot be inspected. `pkfndW.p` implements the Crocker-Grier particle tracking algorithm.

## MATLAB Dependencies

- **MATLAB 2021b** (or later)
- **Image Processing Toolbox** — image operations, filtering, morphological operations
- **Deep Learning Toolbox** — DeepLab v3+ training and inference
- **ResNet50 pretrained model** — backbone for segmentation network
- **Openslide** (external) — reading whole slide image formats (.svs, .ndpi, .tiff)
- **Aperio ImageScope** (external) — manual annotation of training regions (XML export)

## Data Requirements

### Tissue Preparation
- **Fixation**: Standard formalin-fixed, paraffin-embedded (FFPE). Archived tissue blocks work.
- **Sectioning**: Exhaustive serial sectioning at **4 µm thickness**
- **Staining**: Standard H&E on **every 2nd or 3rd section**:
  - Every 3rd section (12 µm effective z-spacing) was used in the original CODA pancreas study and validated to maintain >95% registration quality with <5% cell quantification error vs consecutive sections
  - Every 2nd section (8 µm z-spacing) was used in the fallopian tube study for higher z-resolution
- **Unstained sections**: Mount on plus slides, store at -20°C with desiccant packets under vacuum. These can later be used for IHC, spatial transcriptomics, or other molecular profiling.
- **Slide scanning**: 20x magnification (~0.5 µm/pixel) on a whole slide scanner. The Kiemen lab uses a **Hamamatsu Nanozoomer S210**, saving as .ndpi files (converted to TIFF for processing). Other scanners producing .svs, .ndpi, or BigTIFF formats are compatible.

### Typical Dataset Scale

| Tissue | Sections per specimen | Block dimensions | Cells detected |
|---|---|---|---|
| Pancreas (CODA, 2022) | 101–1,373 (every 3rd stained) | ~2.5 × 2.3 × 0.5 cm | Up to 1.6 billion |
| Fallopian tube (Forjaz et al., 2025) | 601–1,373 (every 2nd stained) | ~2.3 × 2.1 × 0.5 cm | 178–553 million |
| Pancreas immune (Crawford et al., 2024) | ~100+ per sample (48 samples, every 3rd stained) | cm³ scale | Not specified per sample |

### Recommended Pilot Protocol

For a single ~1 cm × 1 cm FFPE tissue block (e.g., tumor biopsy or resection):

| Parameter | Recommendation |
|---|---|
| **Depth to section** | ~1.5 mm |
| **Total sections cut** | ~375 (at 4 µm) |
| **Staining** | H&E every 2nd section → **~188 H&E slides** |
| **Unstained sections** | ~187, mounted on plus slides, stored at -20°C |
| **Scanning** | All H&E slides at 20x (~0.5 µm/pixel) |
| **Effective z-spacing** | 8 µm |
| **Expected file size** | ~1–3 GB per WSI; ~200–550 GB total |

**Why every 2nd section (8 µm) for the pilot**: Registration depends on visual similarity between adjacent sections. At 8 µm spacing, tissue morphology changes very little between frames, making registration robust. This is the safer choice for initial pipeline validation — if registration issues arise, you can be confident it's a software problem, not insufficient z-overlap. Every 3rd section (12 µm) is validated to work well and is fine for production runs once the pipeline is proven; you can computationally skip sections from this pilot dataset to test 12 µm spacing without re-cutting.

**Why ~1.5 mm depth**: Provides enough z-extent to capture real 3D tissue architecture (tumor nests, vessel networks, stromal boundaries) and test computational scalability (~188 WSIs is non-trivial), while keeping costs manageable (~1 day histotech time, ~$200–400 for staining, ~1–2 hours scanning).

**Practical tip**: If the tissue is heterogeneous (e.g., tumor at one end, normal at the other), orient sectioning from the tumor margin inward so the pilot captures the tumor-stroma interface — the most architecturally interesting region.

### For IHC Integration (Optional but Recommended)
Intervening unstained sections can be stained with IHC markers:
- **Immune profiling**: CD45 (pan-leukocyte), CD3/FOXP3 (T cells/Tregs) — as in Crawford et al.
- **Precancer detection**: p53 and Ki67 — as in Forjaz et al.
- **Schedule**: 1 in 8 sections per marker is sufficient for 3D mapping
- IHC sections are co-registered to the H&E stack after the fact (validated workflow: register H&E first, then align IHC to registered H&E)

### Annotation Requirements for Segmentation Model Training
Per organ type (one-time effort):
- **7 evenly spaced tissue images** manually annotated
- **50 examples of each tissue subtype** per image in Aperio ImageScope (or QuPath)
- Iterative: if >90% precision/recall not achieved, add more annotations
- Pre-trained models for pancreas, lung, skin, and liver exist (MATLAB); would need retraining in PyTorch

## CODA Pipeline Evolution

The original CODA pipeline (Kiemen et al., 2022) has been extended in several subsequent publications:

### Multi-Stain IHC Integration (Crawford et al., 2024)
Extended CODA to co-register H&E with IHC-stained serial sections (CD45, CD3/FOXP3). Key developments:
- Validated that **H&E-first registration followed by IHC-to-H&E alignment** is optimal (tested 3 approaches, validated with 100 fiducial landmarks on 50 image pairs)
- **K-medoids clustering** for unsupervised IHC cell classification (positive/negative) from color-deconvolved antibody channels — avoids manual thresholding
- **Power-law model** to estimate 3D immune density from H&E stromal cellularity alone (calibrated on paired H&E + IHC samples, applied to H&E-only cohort)
- **Surface-tangential analysis**: measured immune density change along PanIN surfaces in 12 µm steps; computed decorrelation length scales
- **Immune hotspot/coldspot detection**: algorithmically identified spatial extremes with 0.5 mm minimum separation, exported high-res image patches for pathologist review
- Profiled 1,476 individual PanINs with per-lesion immune metrics, stratified by lesion size
- Demonstrated that 2D cross-sections dramatically over/underestimate true 3D immune density

### Multi-Stain + Hierarchical Segmentation (Forjaz et al., 2025)
Extended CODA for whole-organ 3D reconstruction of fallopian tubes with multi-omic integration:
- **Three cascaded segmentation models**: (1) microenvironment (epithelium, mesothelium, vessels, stroma, fat, nerve, rete ovarii), (2) epithelial subtyping (secretory vs ciliated), (3) IHC signal detection (p53+, Ki67+)
- **StarDist for nuclear detection** (fine-tuned on 25 annotated 256×256 H&E tiles), replacing the original Crocker-Grier particle tracking — extracted 2.19 billion nuclear segmentations
- **Lumen skeletonization and virtual re-sectioning**: computed center path along tube axis, generated up to 10,255 virtual cross-sections perpendicular to the path for compositional analysis
- **Digital SEE-FIM simulation**: virtual subsampling to evaluate clinical protocol sensitivity — showed standard protocols miss >50% of precancerous lesions
- **Lesion growth modeling**: power-law fits to lesion volume distributions
- **InterpolAI** used for missing section restoration in the production pipeline
- Multi-omic integration: 25-plex CODEX proteomics, SRS metabolomics, Visium CytAssist spatial transcriptomics — all guided by 3D-mapped lesion locations

### InterpolAI (Kiemen et al., 2025)
Deep learning optical flow interpolation between serial sections to improve z-resolution and restore microanatomical connectivity. Slots between registration and volume construction.

## Data Flow Summary

```
INPUT
  └── Whole slide images (.svs/.ndpi), one per serial section
       │
       ├── create_downsampled_tif_images.m
       │     └── Downsampled .tif images
       │
       ├── calculate_tissue_ws.m
       │     └── Binary tissue masks
       │
       ├── deconvolve_histological_images.m
       │     └── Normalized H&E + separated hematoxylin channel
       │
       ├── calculate_image_registration.m
       │     └── Registration transforms (rigid + elastic)
       │
       ├── apply_image_registration.m
       │     └── Registered image stack
       │
       ├── train_image_segmentation.m (one-time per organ)
       │     └── Trained DeepLab model (.mat)
       │
       ├── [Inference with trained model]
       │     └── Per-section label maps
       │
       ├── cell_detection.m (on hematoxylin channel)
       │     └── Per-section nuclear coordinates
       │
       ├── register_cell_coordinates.m
       │     └── Registered nuclear coordinates
       │
       ├── build_tissue_volume.m
       │     └── 3D tissue label matrix (12 µm voxels)
       │
       ├── build_cell_volume.m
       │     └── 3D nuclear coordinate matrix
       │
OUTPUT
  ├── 3D tissue volume (labeled)
  ├── 3D cell coordinate volume
  └── 3D visualizations (isosurface renderings)
```
