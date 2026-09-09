"""Semantic segmentation inference using a frozen UNI2-h encoder + DPT decoder.

This is the clean inference entry point for the package: UNI2-h ViT-H features
fused across all four encoder tap points via a Dense Prediction Transformer
(DPT) decoder into per-pixel tissue class logits. DPT is the production
decoder; a linear-probe decoder (single 1x1 conv on the deepest tap only) is
retained alongside it as a fast baseline. Training code for these and other
decoder variants (FPN, Mask2Former) lives in the separate `segmentation_helpers/`
experimentation area, which imports the classes and functions below rather
than redefining them.

Deliberate algorithmic divergence from MATLAB CODA: CODA uses DeepLab v3+ with
a ResNet50 backbone (ImageNet pretrained). path3d uses UNI2-h (ViT-H, pathology
foundation model) + a DPT decoder (Ranftl et al. 2021). Expected benefit:
strong Dice with substantially fewer labeled patches, since the frozen encoder
already carries pathology-specific representations, plus finer class-boundary
localisation than a linear probe from DPT's learned multi-scale fusion.

Why DPT replaced the linear probe as the production decoder: the linear probe
uses only the deepest encoder tap (features[-1], a 32x32 token grid) through a
single 1x1 conv, then two chained bilinear upsamples to reach 448x448. That
structurally caps class-boundary precision at roughly a 32 px block in the
final label map (~16 um at 0.5 um/px), regardless of training quality. DPT
instead reassembles all four encoder taps into a real spatial pyramid
(16 -> 32 -> 64 -> 128 -> 256) via learned Reassemble + RefineNet-style fusion
blocks before the final upsample, giving far finer boundary localisation.

Input convention : (H, W, 3) uint8 RGB tiles at 0.5 um/px.
Output convention: (H, W) int32 label map, values in [0, num_classes).
                   Class 0 = space; tissue classes start at 1.

HuggingFace access
------------------
UNI2-h weights are gated (CC-BY-NC). Each user must run::

    huggingface-cli login

and request access at https://huggingface.co/MahmoodLab/UNI2-h once.
The trained *decoder* checkpoint (~5 MB) is freely shareable -- users working
on similar tissue can load a pre-trained decoder and skip training entirely.
"""

from __future__ import annotations

import gc
import os
import tempfile
from pathlib import Path
from typing import Iterator, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import huggingface_hub

import path3d.config as cfg
from path3d.slide_io import open_slide
from path3d.utils import quiet_progress

# Constants

_UNI2_REPO_ID    = 'MahmoodLab/UNI2-h'
_UNI2_BASE_ARCH  = 'vit_giant_patch14_reg4_dinov2'  # same embed_dim/MLP; override depth+reg_tokens
_UNI2_MEAN       = (0.485, 0.456, 0.406)  # from UNI2 HuggingFace model card
_UNI2_STD        = (0.229, 0.224, 0.225)  # mean and std is to normalize the RGB values from 0-255 to 0-1
_ENCODER_BLOCKS  = [5, 11, 17, 23]        # 0-indexed tap points in UNI2-h (24 blocks)
_SEG_IMG_SIZE    = 448                    # 32x14 -> 32x32 token grid
_ENCODER_DIM     = 1536                   # UNI2-h hidden dim
_DPT_CHANNELS    = 256                    # DPT decoder feature dim; must stay 256 to match
                                           # segmentation_helpers/segmentation_base.py's
                                           # _FPN_CHANNELS, which existing dpt_best.pt
                                           # checkpoints were trained with


# Internal transforms

# Inference: resize tile to encoder input size, then normalize.
_preprocess_tile = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((_SEG_IMG_SIZE, _SEG_IMG_SIZE), antialias=True),
    transforms.Normalize(mean=_UNI2_MEAN, std=_UNI2_STD),
])


def _preprocess(tile: np.ndarray) -> torch.Tensor:
    return cast(torch.Tensor, _preprocess_tile(tile))


def _tile_coords(
    h: int, w: int, tile_size: int, overlap: int,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield (y, x, tile_h, tile_w) for every tile at level-pixel coordinates."""
    stride = tile_size - overlap
    y = 0
    while y < h:
        x = 0
        while x < w:
            yield y, x, min(tile_size, h - y), min(tile_size, w - x)
            x += stride
        y += stride


# Architecture

class SegmentationModel(nn.Module):
    """UNI2-h encoder + decoder for H&E tissue segmentation.

    Base class for all decoder variants (FPN, DPT, LinearProbe, Mask2Former).
    sub classes replace decoder themsleves if present

    Args:
        num_classes: number of tissue classes (class 0 = space)
        freeze_encoder: freeze UNI2-h weights so only the decoder gets trained
    """

    def __init__(self, num_classes: int, freeze_encoder: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self._tok_size = _SEG_IMG_SIZE // 14  # 32 for 448x448 input

        ckpt_path = huggingface_hub.hf_hub_download(
            repo_id=_UNI2_REPO_ID, filename='pytorch_model.bin',
        )
        self.encoder = timm.create_model(
            _UNI2_BASE_ARCH,
            pretrained=False,
            num_classes=0,
            depth=24,
            reg_tokens=8,
            img_size=224,          # checkpoint pos_embed is 16x16 = 256 tokens
            dynamic_img_size=True, # interpolate pos_embed at runtime for 448x448 inference
        )
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        self.encoder.load_state_dict(state_dict, strict=True)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Subclasses overwrite this with their own decoder
        self.decoder: nn.Module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, _SEG_IMG_SIZE, _SEG_IMG_SIZE) normalized float32 tensor.

        Returns:
            (B, num_classes, _SEG_IMG_SIZE, _SEG_IMG_SIZE) class logits.
        """
        B = x.shape[0]
        tok = self._tok_size

        # list of (B, N_tok, C), one per tap point; CLS/register tokens excluded
        raw = self.encoder.forward_intermediates(  # type: ignore
            x,
            indices=_ENCODER_BLOCKS,
            return_prefix_tokens=False,
            output_fmt='NLC',
            intermediates_only=True,
        )

        # (B, N_tok, C) -> (B, C, H_tok, W_tok)
        features = [
            f.reshape(B, tok, tok, _ENCODER_DIM).permute(0, 3, 1, 2).contiguous()
            for f in raw
        ]

        return self.decoder(features)


class LinearProbeDecoder(nn.Module):
    """Single 1x1 conv + bilinear upsample decoder for ViT encoder features

    Minimal linear probe baseline: uses only the deepest encoder tap point (features[-1]) 

    output resolution is derived from the token grid size: for a 448x448
    encoder input with patch size 14, the token grid is 32x32, so the output
    is 32*14 = 448 pixels on each side

    Args:
        in_channels: encoder hidden dim (1536 for UNI2-h)
        num_classes: number of output tissue classes
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Project deepest ViT features to per-pixel class logits.

        Args:
            features: list of 4 tensors (B, in_channels, H_tok, W_tok),
                      ordered shallow to deep. Only features[-1] is used

        Returns:
            (B, num_classes, H_in, W_in) logits at full encoder input resolution,
            where H_in = H_tok * 14 and W_in = W_tok * 14.
        """
        x = self.head(features[-1])           # deepest tap: (B, num_classes, 32, 32)
        h_out = features[-1].shape[-2] * 14   # 32 * 14 = 448
        w_out = features[-1].shape[-1] * 14
        return F.interpolate(x, size=(h_out, w_out), mode='bilinear', align_corners=False)


class LinearProbeSegmentationModel(SegmentationModel):
    """UNI2-h encoder + linear probe decoder for H&E tissue segmentation.

    Subclasses SegmentationModel: the encoder initialisation, forward pass,
    and all training/inference functions are unchanged. Only the
    decoder (self.decoder) is replaced with LinearProbeDecoder / decoder of choice

    Args:
        num_classes: number of tissue classes including space (class 0).
        freeze_encoder: freeze UNI2-h weights so only the linear probe gets trained 
    """

    def __init__(self, num_classes: int, freeze_encoder: bool = True) -> None:
        super().__init__(num_classes=num_classes, freeze_encoder=freeze_encoder)
        self.decoder = LinearProbeDecoder(
            in_channels=_ENCODER_DIM,
            num_classes=num_classes,
        )

# Model construction

def build_linear_probe_model(num_classes: int, freeze_encoder: bool = True) -> nn.Module:
    """Build the UNI2-h + linear probe segmentation model

    Args:
        num_classes: number of tissue classes including space (class 0).
        freeze_encoder: freeze UNI2-h weights so only the linear probe trains.

    Returns:
        Untrained LinearProbeSegmentationModel
    """
    return LinearProbeSegmentationModel(num_classes=num_classes, freeze_encoder=freeze_encoder)


def load_linear_probe_model(path: str | Path, num_classes: int) -> nn.Module:
    """Load a saved linear probe segmentation model.

    Args:
        path: .pt checkpoint file
        num_classes: must match the value used when the model was trained.

    Returns:
        LinearProbeSegmentationModel with loaded weights, in eval mode

    Raises:
        ValueError: if the saved num_classes does not match the requested value
    """
    payload = torch.load(path, map_location='cpu', weights_only=True)
    saved_n = payload.get('num_classes', num_classes)
    if saved_n != num_classes:
        raise ValueError(
            f'num_classes mismatch: file has {saved_n}, caller passed {num_classes}',
        )
    model = LinearProbeSegmentationModel(num_classes=saved_n)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model


# DPT decoder components

class _ResConvUnit(nn.Module):
    """Residual unit: two 3x3 convs with a skip connection (no BN)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _DPTReassemble(nn.Module):
    """Project a ViT feature map (B, C_enc, H_tok, W_tok) to a target scale.

    Args:
        in_channels: encoder hidden dim.
        out_channels: decoder feature dim.
        scale_factor: spatial multiplier relative to the token grid.
            4 or 2 -> ConvTranspose upsample; 1 -> identity; 0.5 -> strided conv.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: float) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if scale_factor == 4:
            self.scale: nn.Module = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=4, stride=4,
            )
        elif scale_factor == 2:
            self.scale = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=2, stride=2,
            )
        elif scale_factor == 1:
            self.scale = nn.Identity()
        else:  # 0.5 -- 2x spatial downsample
            self.scale = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale(self.proj(x))


class _DPTFusionBlock(nn.Module):
    """One stage of DPT RefineNet-style fusion.

    Called as ``fusion(current, prev=None)``.
    For the deepest stage ``prev`` is None (no skip from a deeper level).

    Computes:
        out = outRes( current + skipRes(prev) )  <- skip only when prev given
        out = bilinear_upsample_2x(out)
        out = out_conv(out)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.skip_res = _ResConvUnit(channels)
        self.out_res  = _ResConvUnit(channels)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, prev: torch.Tensor | None = None
    ) -> torch.Tensor:
        if prev is not None:
            x = x + self.skip_res(prev)
        x = self.out_res(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.out_conv(x)


class DPTDecoder(nn.Module):
    """Dense Prediction Transformer decoder for a frozen ViT encoder.

    Creates a genuine multi-scale feature pyramid from 4 ViT tap points
    (all at the same 32x32 token grid resolution) via learned Reassemble
    blocks, then fuses them top-down with RefineNet blocks.

    Reassemble output sizes for a 32x32 token grid (448x448 input, patch=14):
        features[0] (block  5, shallow)  ->  128x128  (4x upsample)
        features[1] (block 11)           ->   64x64   (2x upsample)
        features[2] (block 17)           ->   32x32   (identity)
        features[3] (block 23, deep)     ->   16x16   (2x downsample)

    Fusion cascade (deepest -> shallowest, each stage 2x upsample):
        16 -> 32 -> 64 -> 128 -> 256

    Head: bilinear 256x256 -> 448x448, then 3x3 + 1x1 conv to num_classes.

    Args:
        in_channels: encoder hidden dim (1536 for UNI2-h).
        out_channels: decoder feature dim (256).
        num_classes: number of output tissue classes.
    """

    # Scale factors for Reassemble blocks, ordered shallow -> deep
    _SCALES: list[float] = [4.0, 2.0, 1.0, 0.5]

    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super().__init__()
        self.reassemble = nn.ModuleList([
            _DPTReassemble(in_channels, out_channels, s) for s in self._SCALES
        ])
        self.fusions = nn.ModuleList([
            _DPTFusionBlock(out_channels) for _ in range(4)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, num_classes, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Fuse multi-depth ViT features into class logits via DPT.

        Args:
            features: list of 4 tensors (B, in_channels, H_tok, W_tok),
                      ordered shallow to deep (same format as LinearProbeDecoder).

        Returns:
            (B, num_classes, H_in, W_in) logits at full encoder input resolution.
        """
        h_in = features[0].shape[-2] * 14  # 32 x 14 = 448
        w_in = features[0].shape[-1] * 14

        # Reassemble each tap to its target scale
        r = [ra(f) for ra, f in zip(self.reassemble, features)]
        # r[0] = 128x128, r[1] = 64x64, r[2] = 32x32, r[3] = 16x16

        # Fuse from deepest (16x16) to shallowest (128x128); each stage 2x upsample
        out = self.fusions[3](r[3])               # 16  -> 32
        out = self.fusions[2](r[2], out)          # 32  -> 64
        out = self.fusions[1](r[1], out)          # 64  -> 128
        out = self.fusions[0](r[0], out)          # 128 -> 256

        out = F.interpolate(out, size=(h_in, w_in), mode='bilinear', align_corners=False)
        return self.head(out)


class DPTSegmentationModel(SegmentationModel):
    """UNI2-h encoder + DPT decoder for H&E tissue segmentation.

    Subclasses SegmentationModel: the encoder initialisation, forward pass,
    and all training/inference functions are inherited unchanged. Only the
    decoder (self.decoder) is replaced with a DPTDecoder.

    Args:
        num_classes: number of tissue classes including space (class 0).
        freeze_encoder: freeze UNI2-h weights so only the DPT decoder trains.
    """

    def __init__(self, num_classes: int, freeze_encoder: bool = True) -> None:
        super().__init__(num_classes=num_classes, freeze_encoder=freeze_encoder)
        # The parent leaves self.decoder unassigned; set it to a DPT decoder.
        self.decoder = DPTDecoder(
            in_channels=_ENCODER_DIM,
            out_channels=_DPT_CHANNELS,
            num_classes=num_classes,
        )


# Model construction (DPT)

def build_dpt_model(num_classes: int, freeze_encoder: bool = True) -> nn.Module:
    """Build the UNI2-h + DPT segmentation model.

    Args:
        num_classes: number of tissue classes including space (class 0).
        freeze_encoder: freeze UNI2-h weights so only the DPT decoder trains.

    Returns:
        Untrained DPTSegmentationModel.
    """
    return DPTSegmentationModel(num_classes=num_classes, freeze_encoder=freeze_encoder)


def save_dpt_model(model: nn.Module, path: str | Path) -> None:
    """Save DPT model weights and metadata to a .pt file.

    Args:
        model: a DPTSegmentationModel instance.
        path: destination .pt file.

    Raises:
        TypeError: if model is not a DPTSegmentationModel.
    """
    if not isinstance(model, DPTSegmentationModel):
        raise TypeError(f'Expected DPTSegmentationModel, got {type(model).__name__}')
    torch.save(
        {'state_dict': model.state_dict(), 'num_classes': model.num_classes},
        path,
    )


def load_dpt_model(path: str | Path, num_classes: int) -> nn.Module:
    """Load a saved DPT segmentation model.

    Args:
        path: .pt file written by save_dpt_model().
        num_classes: must match the value used when the model was trained.

    Returns:
        DPTSegmentationModel with loaded weights, in eval mode, on CPU.

    Raises:
        ValueError: if the saved num_classes does not match the requested value.
    """
    payload = torch.load(path, map_location='cpu', weights_only=True)
    saved_n = payload.get('num_classes', num_classes)
    if saved_n != num_classes:
        raise ValueError(
            f'num_classes mismatch: file has {saved_n}, caller passed {num_classes}',
        )
    model = DPTSegmentationModel(num_classes=saved_n)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model


# INFERENCE


def _run_batch(
    tiles: list[np.ndarray],
    positions: list[tuple[int, int, int, int]],
    model: nn.Module,
    accum: np.ndarray,
    count: np.ndarray,
    device: str,
    tile_size: int,
) -> None:
    """Run one batch of tiles through the model; accumulate softmax probabilities."""
    batch = torch.stack([_preprocess(t) for t in tiles]).to(device)
    with torch.inference_mode():
        logits = model(batch)                           # (B, C, 448, 448)
        probs  = torch.softmax(logits, dim=1)
        probs  = F.interpolate(                         # (B, C, tile_size, tile_size)
            probs, size=(tile_size, tile_size),
            mode='bilinear', align_corners=False,
        )
        probs_np = probs.cpu().numpy()

    for prob, (y, x, th, tw) in zip(probs_np, positions):
        accum[:, y:y + th, x:x + tw] += prob[:, :th, :tw]
        count[y:y + th, x:x + tw]    += 1.0


def predict_section(
    path: str | Path,
    model: nn.Module,
    tissue_mask: np.ndarray,
    num_classes: int,
    *,
    section_idx: int,
    seg_mpp: float = cfg.SEG_MPP,
    tile_size: int = 1024,
    overlap: int = 192,
    batch_size: int = 8,
    device: str = 'cuda',
    section_thickness_um: float = 12.0,
    sdata=None,
    label_key_pattern: str = "seg_{:04d}",
    coord_system: str = "microns_3d",
) -> np.ndarray:
    """Run segmentation inference on a registered section.

    Opens the OME-TIFF produced by pipeline.run_pipeline via ``open_slide`` so only one tile
    is in RAM at a time. This file is written at seg_mpp resolution

    Args:
        path: path to the registered OME-TIFF
        model: trained SegmentationModel in eval mode
        tissue_mask: (H, W) bool array. True = background (bright),
            False = tissue (stained) -- matches make_tissue_mask() output.
            Must match the dimensions of the file's best level for
            ``seg_mpp`` -- resize beforehand if needed.
        num_classes: number of tissue classes (must match model).
        section_idx: z-index of this section (used for z placement in sdata).
        seg_mpp: target resolution in um/px (default `config.SEG_MPP`, matching
            tile extraction).
        tile_size: tile side length in pixels.
        overlap: overlap between adjacent tiles in pixels
        batch_size: tiles per GPU batch
        device: 'cuda', 'mps', or'cpu'.
        section_thickness_um: z spacing between sections in microns.
        sdata: optional SpatialData container
        label_key_pattern: format string for the label key in sdata.labels.
        coord_system: target SpatialData coordinate system name.

    Returns:
        (H, W) int32 label map at seg_mpp um/px. (0.5)
        Space pixels are assigned class 0
    """
    model = model.to(device)
    model.eval()

    with open_slide(path) as slide:
        level            = slide.best_level_for_mpp(seg_mpp)
        ds               = slide.level_downsamples[level]
        level_w, level_h = slide.level_dimensions[level]

        # resize tissue_mask if it doesn't already match this level
        if tissue_mask.shape != (level_h, level_w):
            from skimage.transform import resize as sk_resize
            tissue_mask_l = sk_resize(
                tissue_mask.astype(np.float32),
                (level_h, level_w),
                order=0,             # nearest-neighbour. categorical data
                anti_aliasing=False,
            ).astype(bool)
        else:
            tissue_mask_l = tissue_mask

        # make_tissue_mask() returns True=background. invert so True=tissue
        tissue_mask_l = ~tissue_mask_l

        # Use memory-mapped temp files so that accum (num_classes × H × W float32)
        # and count (H × W float32) never need to fit in RAM.  For a typical large
        # section (~82 GB at 6 classes, 0.5 µm/px) this is mandatory to avoid OOM.
        # _run_batch's slice-based "+=" writes work identically on memmaps.
        #
        # On Slurm HPC nodes /tmp is often a small RAM-backed tmpfs; use
        # $SLURM_TMPDIR (per-job local SSD scratch) when available.
        _scratch  = os.environ.get("SLURM_TMPDIR") or None
        _tmp_dir  = Path(tempfile.mkdtemp(prefix="path3d_seg_", dir=_scratch))
        _acc_path = _tmp_dir / "accum.dat"
        _cnt_path = _tmp_dir / "count.dat"
        accum = np.memmap(str(_acc_path), dtype=np.float32, mode="w+",
                          shape=(num_classes, level_h, level_w))
        count = np.memmap(str(_cnt_path), dtype=np.float32, mode="w+",
                          shape=(level_h, level_w))

        batch_tiles:     list[np.ndarray]                = []
        batch_positions: list[tuple[int, int, int, int]] = []

        for y, x, th, tw in quiet_progress(
            list(_tile_coords(level_h, level_w, tile_size, overlap)),
            desc='Segmenting tiles',
        ):
            if tissue_mask_l[y:y + th, x:x + tw].mean() < 0.1:
                # space tile - assign full weight to class 0, skip network
                accum[0, y:y + th, x:x + tw] += 1.0
                count[y:y + th, x:x + tw]    += 1.0
                continue

            tile = slide.read_region(
                (int(x * ds), int(y * ds)), level, (tw, th),
            )  # (th, tw, 3) uint8

            # pad boundary tiles to tile_size for uniform batch dimensions
            if th < tile_size or tw < tile_size:
                padded           = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:th, :tw] = tile
                tile             = padded

            batch_tiles.append(tile)
            batch_positions.append((y, x, th, tw))

            if len(batch_tiles) == batch_size:
                _run_batch(batch_tiles, batch_positions, model, accum, count, device, tile_size)
                batch_tiles.clear()
                batch_positions.clear()

        if batch_tiles:
            _run_batch(batch_tiles, batch_positions, model, accum, count, device, tile_size)

    # average overlapping predictions and derive label map in row chunks so that
    # the full accum array is never brought into RAM at once.
    _CHUNK = 512  # rows per chunk; tune down if not enough scratch disk space
    label_map = np.zeros((level_h, level_w), dtype=np.int32)
    for _r0 in range(0, level_h, _CHUNK):
        _r1 = min(_r0 + _CHUNK, level_h)
        _ca = np.array(accum[:, _r0:_r1, :], dtype=np.float32)   # (C, chunk, W) in RAM
        _cc = np.array(count[_r0:_r1, :],    dtype=np.float32)   # (chunk, W)
        _ca /= np.maximum(_cc[np.newaxis], 1.0)
        _nm  = ~tissue_mask_l[_r0:_r1, :]                        # non-tissue mask for this chunk
        _ca[:,  _nm] = 0.0
        _ca[0,  _nm] = 1.0
        label_map[_r0:_r1, :] = np.argmax(_ca, axis=0).astype(np.int32)
        del _ca, _cc

    # release memmaps and remove temp files
    del accum, count
    gc.collect()
    _acc_path.unlink(missing_ok=True)
    _cnt_path.unlink(missing_ok=True)
    try:
        _tmp_dir.rmdir()
    except OSError:
        pass

    if sdata is not None:
        _attach_labels_to_sdata(
            sdata, label_map, section_idx,
            seg_mpp=seg_mpp,
            section_thickness_um=section_thickness_um,
            label_key=label_key_pattern.format(section_idx),
            coord_system=coord_system,
        )

    return label_map


def _attach_labels_to_sdata(
    sdata,
    label_map: np.ndarray,
    section_idx: int,
    *,
    seg_mpp: float,
    section_thickness_um: float,
    label_key: str,
    coord_system: str,
) -> None:
    """Store a label map in a SpatialData container tied to microns_3d

    Builds an Affine mapping label-map pixel coordinates (y_px, x_px) to
    physical coordinates (z_um, y_um, x_um) in ``coord_system``.
    same convention as registration

    Raises:
        ImportError: spatialdata is not installed
    """
    try:
        from spatialdata.models import Labels2DModel
        from spatialdata.transformations import Affine, set_transformation  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "spatialdata is required for sdata output. "
            "Install: pip install spatialdata"
        ) from exc

    z_um = section_idx * section_thickness_um

    # (y_px, x_px) -> (z_um, y_um, x_um) - same convention as registration.py
    matrix = np.array([
        [0.,       0.,       z_um   ],   # z is constant for this section
        [seg_mpp,  0.,       0.     ],   # y_um = seg_mpp * y_px
        [0.,       seg_mpp,  0.     ],   # x_um = seg_mpp * x_px
        [0.,       0.,       1.     ],
    ])
    transform = Affine(matrix, input_axes=("y", "x"), output_axes=("z", "y", "x"))

    labels_el = Labels2DModel.parse(
        label_map,
        dims=("y", "x"),
        transformations={coord_system: transform},
    )
    sdata.labels[label_key] = labels_el
