# References

## Primary CODA Publication

- Kiemen AL, Braxton AM, Grahn MP, et al. "CODA: quantitative 3D reconstruction of large tissues at cellular resolution." *Nature Methods* 19, 1490–1499 (2022).
  - DOI: https://doi.org/10.1038/s41592-022-01650-9
  - PMC full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC10500590/
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/36280719/

## CODA Application Studies

These papers apply and extend the CODA pipeline, demonstrating advanced methods that inform the path3d design.

- Crawford JM, Forjaz A, Crawford AC, et al. "Quantitative 3D histology reveals localized immune remodeling during early pancreatic cancer progression." *Cell Press Blue* (2026).
  - DOI: https://doi.org/10.1016/j.xcrm.2026.100006 (approximate — Cell Press URL: https://www.cell.com/cell-press-blue/fulltext/S3051-3839(26)00006-X)
  - PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11326156/
  - **Key methods**: Multi-stain IHC co-registration (CD45, CD3/FOXP3), k-medoids IHC cell classification, power-law H&E-to-immune density transfer model, surface-tangential decorrelation analysis, immune hotspot/coldspot detection, 2D-vs-3D comparison, per-lesion immune profiling of 1,476 PanINs, 38-plex IMC integration

- Forjaz A, Queiroga V, Li Y, et al. "3D multi-omic mapping of whole nondiseased human fallopian tubes at cellular resolution reveals a large incidence of ovarian cancer precursors." *bioRxiv* (2025).
  - DOI: https://doi.org/10.1101/2025.09.21.677628
  - **Key methods**: Three-model cascaded segmentation, StarDist nuclear detection (2.19B nuclei), lumen skeletonization + virtual re-sectioning, digital SEE-FIM subsampling simulation, InterpolAI for missing section restoration, lesion growth modeling, multi-omic integration (25-plex CODEX, SRS metabolomics, Visium CytAssist)

- Kiemen AL, et al. "Combined assembloid modeling and 3D whole-organ mapping captures the microanatomy and function of the human fallopian tube." *Science Advances* 10, eadp6285 (2024).
  - DOI: https://doi.org/10.1126/sciadv.adp6285

## Original CODA Code and Protocols

- **MATLAB source code**: https://github.com/ashleylk/CODA
- **Kiemen Lab CODA methodology page**: https://labs.pathology.jhu.edu/kiemen/coda-3d/
- **HuBMAP protocol (protocols.io)**: https://www.protocols.io/view/coda-3d-tissue-reconstruction-pipeline-hubmap-jhu-db8z2rx6

## Python Libraries — Core Pipeline

### WSI I/O
- **openslide-python**: https://openslide.org/api/python/ — Python bindings for reading whole slide images
- **tiffslide**: https://github.com/Bayer-Group/tiffslide — Pure-Python OpenSlide drop-in replacement
- **tifffile**: https://github.com/cgohlke/tifffile — Read/write TIFF and OME-TIFF

### Image Processing
- **scikit-image**: https://scikit-image.org/ — Image processing (registration, morphology, color, feature detection)
- **OpenCV (cv2)**: https://docs.opencv.org/4.x/ — Computer vision (alternative for image ops)
- **scipy.ndimage**: https://docs.scipy.org/doc/scipy/reference/ndimage.html — N-dimensional image processing

### Registration
- **VALIS**: https://github.com/MathOnco/valis — Serial section WSI registration (rigid + non-rigid)
  - Documentation: https://valis.readthedocs.io/
  - Publication: Gatenbee et al., "Virtual alignment of pathology image series for multi-gigapixel whole slide images." *Nature Communications* 14, 4502 (2023). https://doi.org/10.1038/s41467-023-40218-9
- **SimpleITK**: https://simpleitk.org/ — Medical image registration toolkit

### Stain Normalization
- **torchstain**: https://github.com/EIDOSLAB/torchstain — GPU-accelerated stain normalization (Macenko, Reinhard). **Recommended.**
- **histomicstk**: https://github.com/DigitalSlideArchive/HistomicsTK — Histology analysis toolkit (maintained by Kitware); includes Macenko/Reinhard normalization
- ~~**staintools**: https://github.com/Peter554/StainTools~~ — **Archived May 2021, unmaintained. Do not use.**

### Deep Learning / Segmentation
- **PyTorch**: https://pytorch.org/
- **torchvision segmentation models**: https://pytorch.org/vision/stable/models.html#semantic-segmentation — DeepLab v3 (not v3+) ResNet50 included
- **segmentation_models_pytorch (smp)**: https://github.com/qubvel-org/segmentation_models.pytorch — Proper DeepLab v3+, UNet, UNet++ with various encoder backbones
- **albumentations**: https://albumentations.ai/ — Image augmentation for training

### Nuclear Detection
- **trackpy**: https://github.com/soft-matter/trackpy — Python implementation of Crocker-Grier particle tracking. **Direct equivalent of CODA's `pkfndW` function.** Recommended for faithful porting.
  - Documentation: https://soft-matter.github.io/trackpy/
- **Cellpose**: https://github.com/MouseLand/cellpose — Generalist cell segmentation. **PyTorch-native (no TensorFlow dependency).** Recommended for Tier 2 upgrade.
  - Publication: Stringer et al., "Cellpose: a generalist algorithm for cellular segmentation." *Nature Methods* 18, 100–106 (2021).
- **StarDist**: https://github.com/stardist/stardist — Star-convex object detection, pre-trained H&E model available. Note: requires TensorFlow (potential dependency conflict).
  - Publication: Schmidt et al., "Cell Detection with Star-Convex Polygons." *MICCAI* (2018).

### 3D Visualization
- **napari**: https://napari.org/ — Interactive multi-dimensional image viewer
- **pyvista**: https://docs.pyvista.org/ — 3D plotting and mesh analysis
- **vedo**: https://vedo.embl.es/ — 3D scientific visualization

### Large Data Handling
- **dask**: https://dask.org/ — Parallel and out-of-core array computation
- **zarr**: https://zarr.readthedocs.io/ — Chunked, compressed N-dimensional arrays
- **cucim** (RAPIDS): https://github.com/rapidsai/cucim — GPU-accelerated image I/O and processing

## Related Tools and Methods

### 3D Histology Reconstruction (Alternative Approaches)
- **TriPath**: https://github.com/mahmoodlab/TriPath — Weakly supervised analysis of 3D pathology samples
- **Path2MR**: https://github.com/ortegacruzd/Path2MR — 3D histology reconstruction without MRI reference

### Spatial Transcriptomics Integration
- **squidpy**: https://squidpy.readthedocs.io/ — Spatial single-cell analysis (includes Cellpose integration)
- **scanpy**: https://scanpy.readthedocs.io/ — Single-cell analysis
- **anndata**: https://anndata.readthedocs.io/ — Annotated data matrices
- **spatialdata**: https://spatialdata.scverse.org/ — Unified framework for spatial omics data
- **inferCNV**: https://github.com/broadinstitute/inferCNV — Copy number variation inference from scRNA-seq / spatial transcriptomics

### Multiplexed Proteomics
- **CODEX / Akoya PhenoCycler**: Cyclic immunofluorescence for highly multiplexed protein imaging
- **Imaging Mass Cytometry (IMC)**: Standard BioTools Hyperion platform for 40+ marker metal-tagged antibody panels

### Pathology Foundation Models (Stretch)
- **UNI**: https://github.com/mahmoodlab/UNI — Vision transformer pre-trained on 100M+ histology patches
- **Virchow**: Meta pathology foundation model
- **Phikon**: https://github.com/owkin/HistoSSLscaling — Self-supervised histology model
- **CONCH**: https://github.com/mahmoodlab/CONCH — Contrastive learning for histology

### Annotation Tools
- **QuPath**: https://qupath.github.io/ — Open-source digital pathology (alternative to Aperio ImageScope for annotations)

## Follow-Up Publications

- Kiemen AL, et al. "InterpolAI: deep learning-based optical flow interpolation and restoration of biomedical images for improved 3D tissue mapping." *Nature Methods* (2025). https://doi.org/10.1038/s41592-025-02712-4
  - Extension of CODA with deep learning interpolation between sections
  - Now confirmed as a production component of the Kiemen lab pipeline (used in Forjaz et al. 2025)

See also **CODA Application Studies** at the top of this document for the two most recent papers that extend the CODA pipeline with multi-stain IHC, immune profiling, virtual re-sectioning, and multi-omic integration.
