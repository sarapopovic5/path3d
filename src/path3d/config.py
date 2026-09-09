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
