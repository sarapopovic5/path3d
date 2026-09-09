"""Downsampling, tissue masking, stain normalization, and section QC.

All functions operate on numpy arrays (H, W, 3) uint8 RGB unless noted.
Use slide_io.read_full_level / iter_tiles to get arrays from a WSI first.
"""

from __future__ import annotations 
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
from skimage.transform import rescale
from skimage.color import rgb2gray

from torchstain.torch.normalizers.macenko import TorchMacenkoNormalizer
from torchvision import transforms

from path3d import config as cfg
from path3d.slide_io import open_slide
from path3d.slide_io import SlideReader


# Downsampling


def _block_mean_tiled(
    slide: SlideReader,
    level: int,
    factor: int,
    tile_target: int = 4096,
) -> np.ndarray:
    """Downsample a whole pyramid level by an integer ``factor``, tile by tile.

    Reads ``level`` in tiles and block-averages each one, so peak memory is
    one tile plus the (much smaller) output -- never the whole level. This is
    what makes ``downsample_to_mpp`` safe on slides whose pyramid is missing
    or unusable: VALIS-written registered OME-TIFFs expose a single level to
    OpenSlide (``n_levels == 1``, confirmed on real output), so the "best
    level" for a coarse target is full resolution. Materialising that level
    and rescaling it in float64 peaked around 45+ GB for one 802-megapixel
    section and was OOM-killed at 64 GB on HPC.

    Block averaging (rather than per-tile interpolation) is used because tiles
    are aligned to exact multiples of ``factor``, so every output pixel is
    computed from its own disjoint input block. That makes the result
    seam-free and independent of tile size -- per-tile ``rescale`` would
    truncate its anti-aliasing filter at tile edges and leave a visible grid.

    Args:
        slide: Open SlideReader.
        level: Pyramid level to read.
        factor: Integer downsample factor (>= 1) applied to ``level``.
        tile_target: Approximate tile side length in level pixels; rounded
            down to a multiple of ``factor``.

    Returns:
        (H // factor, W // factor, 3) uint8 RGB array. Up to ``factor - 1``
        pixels are trimmed from the right/bottom edge so the level divides
        evenly into whole blocks.
    """
    level_w, level_h = slide.level_dimensions[level]
    ds = slide.level_downsamples[level]

    out_h = max(1, level_h // factor)
    out_w = max(1, level_w // factor)
    out = np.empty((out_h, out_w, 3), dtype=np.uint8)

    # Cropped extent that divides evenly into factor-sized blocks.
    crop_h, crop_w = out_h * factor, out_w * factor
    step = max(factor, (tile_target // factor) * factor)

    for iy in range(0, crop_h, step):
        ih = min(step, crop_h - iy)
        for ix in range(0, crop_w, step):
            iw = min(step, crop_w - ix)

            # read_region takes a level-0 location; size is at `level`.
            tile = slide.read_region(
                (int(ix * ds), int(iy * ds)), level, (iw, ih),
            )  # (ih, iw, 3) uint8

            th, tw = ih // factor, iw // factor
            # float32 (not float64) keeps the per-tile mean buffer half-size.
            block = tile[: th * factor, : tw * factor].reshape(
                th, factor, tw, factor, 3
            )
            reduced = block.mean(axis=(1, 3), dtype=np.float32)
            out[iy // factor : iy // factor + th,
                ix // factor : ix // factor + tw] = reduced.astype(np.uint8)

    return out


def downsample_to_mpp(
    slide: SlideReader,
    target_mpp: float,
) -> np.ndarray:
    """Return a full-slide image at the requested resolution.

    Selects the best pyramid level (the coarsest level whose actual MPP does
    not exceed target_mpp), then reduces it to ``target_mpp``.

    The reduction is done tile-by-tile via ``_block_mean_tiled`` whenever the
    level needs downsampling by a factor >= 2, so the full level is never held
    in RAM and never converted to float. An earlier version read the whole
    level and rescaled it in float64, which OOM-killed a 64 GB HPC job on
    VALIS-registered OME-TIFFs (single-level pyramid -> 802-megapixel "best"
    level -> ~45 GB peak); see ``_block_mean_tiled``. Any leftover
    non-integer residual is applied afterwards, on the already-small array.

    Args:
        slide: Open SlideReader (OpenSlideReader or CziReader).
        target_mpp: Desired resolution in microns per pixel (e.g. 80.0). For our lab's purposes:
        80 µm/pixel: Global rigid registration - lowest resolution / less clear
        8 µm/pixel (REG_MPP): Elastic registration (local tile matching)
        0.5 µm/pixel (SEG_MPP): Semantic segmentation and cell detection - highest resolution / clearest
        12 µm/pixel isotropic is the default 3D voxel size used downstream (volume.py /
        quantification.py's voxel_um / target_voxel_um parameters); it is caller-supplied
        and not a constant baked into this function.

    Returns:
        (H, W, 3) uint8 RGB array at target_mpp
    """
    level = slide.best_level_for_mpp(target_mpp)
    level_w, level_h = slide.level_dimensions[level]
    actual_mpp = slide.get_mpp_at_level(level)

    # How much this level still has to shrink. >= 1 when downsampling;
    # < 1 only when target_mpp is finer than the slide's own level 0.
    factor = target_mpp / actual_mpp
    int_factor = max(1, int(np.floor(factor + 1e-6)))

    if int_factor >= 2:
        img = _block_mean_tiled(slide, level, int_factor)
    else:
        # No integer reduction available -- read the level directly.
        img = slide.read_region((0, 0), level, (level_w, level_h)) # don't need to .convert("RGB"), SlideReader function already returns np array

    # Residual (non-integer) scale left over after the integer reduction,
    # applied to the already-reduced array so this stays cheap.
    residual_scale = (actual_mpp * int_factor) / target_mpp

    if not np.isclose(residual_scale, 1.0, rtol=1e-3):
        img_float = img.astype(np.float32) / 255.0
        img_float = rescale(
            img_float,
            residual_scale,
            anti_aliasing=residual_scale < 1.0, # for downscaling only. False if upscaling
            channel_axis=2,
        )
        img = (img_float * 255.0).clip(0, 255).astype(np.uint8)

    return img




# Tissue masking


def _first_valley_after_wide_hump(
    gray: np.ndarray,
    n_bins: int = 256,
    min_hump_width_frac: float = 0.06,
    min_prominence_frac: float = 0.01,
) -> float | None:
    """Detect the valley between the real-tissue continuum and a padding spike.

    NOTE: make_tissue_mask no longer calls this function -- it now uses a
    fixed grayscale cutoff (cfg.TISSUE_GRAY_THRESHOLD) instead of a per-image
    fitted threshold. This function is retained solely as a diagnostic used
    by segmentation_helpers/inspect_tissue_histogram.py; it is not part of
    the production mask path.

    Real stained tissue produces a broad, smoothly-varying grayscale
    distribution (natural variation in optical density). Manufactured
    background fills -- CZI mosaic scan-padding, or a flat slide-canvas
    color -- are each a single near-uniform color, so they show up as a
    narrow, tall spike rather than a spread distribution. When such a
    padding spike sits close (in gray value) to the true-white background
    spike, a single global 2-class Otsu threshold becomes unstable: its
    variance-minimizing split point can land above or below the padding
    spike depending on the *relative size* of these populations, not their
    values. See .planning/debug/resolved/tissue-mask-otsu-padding.md.

    This looks for that specific trimodal signature directly: the first two
    significant peaks in a smoothed grayscale histogram, and the valley
    between them. It only returns a value when the first peak is wide
    enough to be a genuine continuum (not a single flat fill) -- this makes
    the function a no-op on simple bimodal images (e.g. any single-color
    synthetic tissue patch on a single-color background), where there is no
    natural continuum to speak of.

    Args:
        gray: (H, W) float64 grayscale image in [0, 1].
        n_bins: Number of histogram bins.
        min_hump_width_frac: Minimum width of the first peak, as a fraction
            of n_bins, required to treat it as a natural continuum rather
            than a flat fill.
        min_prominence_frac: Minimum peak prominence, as a fraction of the
            tallest bin, required to count as a significant mode.

    Returns:
        Grayscale value of the detected valley, or None if no confident
        trimodal (continuum + spike) signature is found.
    """
    counts, edges = np.histogram(gray, bins=n_bins, range=(0.0, 1.0))
    smooth = gaussian_filter1d(counts.astype(np.float64), sigma=2)
    if smooth.sum() <= 0:
        return None

    min_prominence = smooth.max() * min_prominence_frac
    peaks, _ = find_peaks(smooth, prominence=min_prominence)
    if len(peaks) < 2:
        return None  # need at least two significant modes to call this trimodal

    first_peak, second_peak = peaks[0], peaks[1]

    widths, *_ = peak_widths(smooth, [first_peak], rel_height=0.9)
    if widths[0] < min_hump_width_frac * n_bins:
        return None  # first mode is a narrow flat fill, not a natural continuum

    valley_idx = first_peak + int(np.argmin(smooth[first_peak : second_peak + 1]))
    valley_height = smooth[valley_idx]
    if valley_height >= smooth[second_peak] * 0.5:
        return None  # not a confident dip relative to the following spike

    return float(edges[valley_idx])


def make_tissue_mask(
    image_rgb: np.ndarray,
    threshold: float = cfg.TISSUE_GRAY_THRESHOLD,
) -> np.ndarray:
    """Return a binary tissue mask using a fixed grayscale cutoff.

    Deliberate algorithmic divergence from MATLAB CODA: this
    replaces both whole-image grayscale Otsu thresholding AND the
    `_first_valley_after_wide_hump` valley-correction patch that used to sit
    on top of it. Otsu's per-image fitted threshold was measured *flipping*
    on real data -- on section 0087 of a 149-section stack it produced a
    precise (non-blob) mask instead of the solid blob every other section
    got. A threshold fitted per-image cannot be relied on to be stable across
    a whole stack, no matter how it is corrected after the fact. The fixed
    cutoff below reproduces each section's own Otsu result to within 1e-5 on
    32/32 sections tested -- it is the *same mask made deterministic*, not a
    different mask.

    Evidence (sweeps on registered OME-TIFFs at 2.0 um/px, pipeline.py's
    _MASK_MPP; see cfg.TISSUE_GRAY_THRESHOLD for the full write-up): the
    grayscale histogram is trimodal -- tissue continuum ~0.02-0.66, a
    ~0.72-0.75 spike, and a true-white spike ~1.0. The ~0.74 spike is REAL
    slide background (glass, lumina, inter-fragment gaps; ~30.9% of a raw
    CZI; spatially interleaved with tissue), NOT scan-edge padding -- the
    actual canvas fill is the >0.97 white (libCZI out-of-FOV + VALIS white
    warp, 61.1% of a registered section). A 0.80 cutoff therefore admits
    real slide background and excludes only the canvas, and that is
    precisely what makes the mask a blob: the gaps BETWEEN tissue fragments
    are 0.74, so including them closes the silhouette instead of tracing
    each fragment. 0.740 fails on every section tested (edge density
    0.086-0.162, ragged) because it cuts through the middle of that
    background population; 0.760-0.860 all produce a solid blob (edge
    density 0.0012-0.0028) on 32/32 sections tested; the [0.76, 0.86]
    corridor holds only 0.005-0.0125% of pixels, so tissue fraction barely
    moves across it. 0.80 sits centrally.

    Design intent -- the mask's job is to SAVE COMPUTE, not to decide
    background:
    - The tile gate at segmentation.py:372 skips tiles under 10% tissue
      coverage.
    - The model itself predicts class 0 ("space") everywhere else.
    - A blob mask has no interior holes, so the pixel override at
      segmentation.py:408-410 -- which hard-sets mask-background pixels to
      class 0 over the model's prediction -- can only ever act on the thin
      band OUTSIDE the blob. The blob mask makes that override harmless by
      construction, which is precisely why segmentation.py needs no change
      for this divergence.
    - Cost of the blob: ~2.4x the true tissue area in tiles inferred.
      Accepted.

    Args:
        image_rgb: (H, W, 3) uint8 RGB array (typically a low-res overview).
        threshold: Grayscale cutoff in [0, 1], defaults to
            cfg.TISSUE_GRAY_THRESHOLD. Callers may override for
            experimentation, but no production call site does.

    Returns:
        (H, W) bool array. Polarity is UNCHANGED from the prior Otsu
        implementation: True = background (bright), False = tissue
        (stained). Every consumer inverts this mask (segmentation.py:347,
        segmentation_helpers/annotation.py:87,
        segmentation_helpers/inference_nomask.py:8) -- do not flip it.
    """
    gray = rgb2gray(image_rgb)
    return gray > threshold


# Stain normalization

def fit_stain_normalizer(
    manifest_path: str | Path,
    mpp: float = 8.0,
) -> TorchMacenkoNormalizer:
    """Fit a Macenko stain normalizer on the middle section of the stack.

    Opens the middle slide once, reads it at the requested resolution, fits
    the normalizer, then closes the slide. Call this once in run_pipeline and
    pass the returned normalizer to apply_stain_normalizer for each section.

    user should manually verify that the middle section has clean H&E staining
    (no folds, tears, or unusual staining) before running the full pipeline.

    Args:
        manifest_path: Path to the manifest CSV from slide_io.create_manifest.
        mpp: Resolution at which to read the reference image (microns/pixel).

    Returns:
        Fitted MacenkoNormalizer ready to be passed to apply_stain_normalizer.
    """
    ref_path = find_middle_image(manifest_path)
    with open_slide(ref_path) as ref_slide:
        level = ref_slide.best_level_for_mpp(mpp)
        w, h = ref_slide.level_dimensions[level]
        reference_rgb = ref_slide.read_region((0, 0), level, (w, h))

    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])
    normalizer = TorchMacenkoNormalizer()
    normalizer.fit(T(reference_rgb))
    return normalizer


def apply_stain_normalizer(
    image_rgb: np.ndarray,
    normalizer: TorchMacenkoNormalizer,
) -> np.ndarray:
    """Apply a pre-fitted Macenko normalizer to an image

    Args:
        image_rgb: (H, W, 3) uint8 RGB array to normalize
        normalizer: Fitted normalizer returned by fit_stain_normalizer

    Returns:
        (H, W, 3) uint8 array, normalized RGB.
    """
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
    ])
    norm, _, _ = normalizer.normalize(I=T(image_rgb), stains=True)
    return norm.clamp(0, 255).byte().numpy()  # type: ignore  # torchstain returns (H, W, C)


def find_middle_image(manifest_path: str | Path) -> str:
    """Return the file path of the middle slide in the ordered H&E stack.

    Used to select a reference image for stain normalization.
    """
    df = pd.read_csv(manifest_path)
    middle_idx = len(df) // 2
    print(f"Middle image is index {middle_idx}. manually check if this image is normally stained (recall manifest is 0 based). if not, pick nearest image which is")
    return str(df.iloc[middle_idx, 2])


