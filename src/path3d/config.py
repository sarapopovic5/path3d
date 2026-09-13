"""Tissue class definitions, colors, and default parameters

TISSUE_CLASSES maps organ name → {class_index: name}. can add multiple tissue types

Also holds the default resolution constants (SEG_MPP, REG_MPP) and the
default tissue-mask grayscale cutoff (TISSUE_GRAY_THRESHOLD) shared across
the pipeline

includes HGSC tissue type only for now, update if training model on new tissue type (ex LGSC, endo)

"""

import numpy as np

# Documented project resolution levels. defined just once here

SEG_MPP = 0.5   # segmentation and nuclear-detection working resolution, um/px
REG_MPP = 8.0   # elastic serial-section registration resolution, um/px

# Fixed grayscale cutoff for preprocessing.make_tissue_mask, in [0, 1] on an
# rgb2gray float image (True = background when gray > threshold). Replaces a
# per-image Otsu fit that was measured flipping to a non-blob (precise) mask
# on one real section (0087) out of a 149-section stack -- a fitted threshold
# cannot be relied on to be stable across the whole stack. Evidence from
# sweeps on registered OME-TIFFs at 2.0 um/px (pipeline.py's _MASK_MPP):
# the grayscale histogram is trimodal (tissue continuum ~0.02-0.66, a
# ~0.72-0.75 spike, a true-white spike ~1.0). The ~0.74 spike is REAL slide
# background -- glass, lumina and inter-fragment gaps, ~30.9% of a raw CZI,
# spatially interleaved with the tissue -- NOT scan-edge padding; the actual
# canvas fill (libCZI out-of-FOV + VALIS white warp) is the >0.97 white,
# 61.1% of a registered section. So 0.80 deliberately admits real slide
# background and excludes only the canvas, which is exactly why the result
# is a solid blob: the gaps BETWEEN tissue fragments are 0.74, so including
# them closes the silhouette. 0.740 fails (ragged, edge density
# 0.086-0.162) on every section tested because it cuts through the middle of
# that background population; 0.760-0.860 all produce a
# solid blob (edge density 0.0012-0.0028, well under the 0.02 blob bar) on
# 32/32 sections tested. Only 0.005-0.0125% of pixels fall in [0.76, 0.86],
# so tissue fraction barely moves (0.09-0.20%) across that whole span --
# 0.80 sits centrally, ~3 sweep steps of margin from the 0.740 failure, and
# reproduces each section's own per-image Otsu result to within 1e-5.
TISSUE_GRAY_THRESHOLD = 0.80

TISSUE_CLASSES = {
    "HGSC" : {
        0: "space",
        1: "Epithelium",   # Order must match CLASS_MAP in scripts/prepare_training_data.py
        2: "Stroma",       
        3: "rbc",
        4: "Intraluminal secretion"
    }
}


TISSUE_CLASS_COLORS = {
    "HGSC": np.array(
        [
            [0, 0, 0],        # 0 space
            [228, 26, 28],    # 1 Epithelium
            [55, 126, 184],   # 2 Stroma
            [77, 175, 74],    # 3 rbc
            [255, 127, 0],    # 4 Intraluminal secretion
        ],
        dtype=np.uint8,
    )
}


# Xenium-derived tissue NEIGHBOURHOODS (path3d.niches) -- a different axis from
# TISSUE_CLASSES above. TISSUE_CLASSES["HGSC"] is a per-pixel morphological
# segmentation (is this pixel epithelium or stroma?); these are 50 um
# neighbourhood classes predicted per 32 um tile (what kind of region is this?).
# Both can be built for the same stack; they are not interchangeable.
#
# Registered as a TISSUE_CLASSES key so view_volume / compute_volumetrics /
# _write_section_outputs work on a niche volume unchanged, via
# tissue_type="HGSC_niches".
#
# Index 0 is background (no prediction: off-tissue, or inside the canvas margin
# a 112 um window cannot fit in), matching Labels3DModel's and napari's
# background convention.
NICHE_BACKGROUND_INDEX = 0

# Class NAME -> label index. Deliberately keyed by name, not by position in the
# classifier's classes_ array: sklearn orders classes_ alphabetically
# (acellular, epithelium, immune, stroma), so a positional convention would
# silently reassign every index if the bundle were retrained with a different
# class set. path3d.niches.rasterize.argmax_labels looks up by name.
NICHE_LABEL_INDEX = {
    "epithelium": 1,
    "immune": 2,
    "stroma": 3,
    "acellular": 4,
}

TISSUE_CLASSES["HGSC_niches"] = {
    NICHE_BACKGROUND_INDEX: "background",
    **{index: name for name, index in NICHE_LABEL_INDEX.items()},
}

# Palette from the niche model card's own figures, so a 3D volume reads the
# same way as the per-slide niche_maps.png QC figure.
TISSUE_CLASS_COLORS["HGSC_niches"] = np.array(
    [
        [0, 0, 0],        # 0 background
        [123, 63, 157],   # 1 epithelium  #7b3f9d
        [201, 85, 63],    # 2 immune      #c9553f
        [63, 127, 157],   # 3 stroma      #3f7f9d
        [184, 176, 164],  # 4 acellular   #b8b0a4
    ],
    dtype=np.uint8,
)
