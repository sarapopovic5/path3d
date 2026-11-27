#!/usr/bin/env python3
"""Macenko stain normalisation to a fixed canonical H&E reference.

Applied to every tile before embedding. QC (63/64) found the TMA test slides absorb >2x the
green-channel OD of every training slide -- a shift 3-4x wider than the spread among training
slides. The decision experiment (65/66, training slides only) showed normalisation is NEUTRAL
on the within-train stain range (LOSO +0.017 bal-acc, divergent pair -0.002, both noise), so
it is applied as insurance against the much larger train->TMA shift it could not measure.
Fixed reference (not per-slide) so train and test map to the same target space.
"""
import numpy as np

REF_STAIN = np.array([[0.5626, 0.7201, 0.4062], [0.2159, 0.8012, 0.5581]])   # H, E
REF_CONC = np.array([1.9705, 1.0308])

def macenko(img, Io=240, alpha=1, beta=0.15):
    """Normalise an RGB uint8 tile to REF_STAIN. Returns uint8; passes the tile through
    unchanged if it has too little stained content to estimate vectors from."""
    h, w, _ = img.shape
    I = img.reshape(-1, 3).astype(np.float32)
    OD = -np.log10(np.clip((I + 1) / Io, 1e-6, None))
    ODhat = OD[~(OD < beta).any(1)]
    if len(ODhat) < 50:
        return img
    try:
        _, V = np.linalg.eigh(np.cov(ODhat.T))
    except np.linalg.LinAlgError:
        return img
    Vt = V[:, 1:3]
    That = ODhat @ Vt
    phi = np.arctan2(That[:, 1], That[:, 0])
    a1, a2 = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    v1 = Vt @ np.array([np.cos(a1), np.sin(a1)])
    v2 = Vt @ np.array([np.cos(a2), np.sin(a2)])
    HE = np.array([v1, v2]) if v1[0] > v2[0] else np.array([v2, v1])
    HE = HE / (np.linalg.norm(HE, axis=1, keepdims=True) + 1e-9)
    C, *_ = np.linalg.lstsq(HE.T, OD.T, rcond=None)
    maxC = np.percentile(C, 99, axis=1)
    C = C * (REF_CONC / np.maximum(maxC, 1e-6))[:, None]
    Inorm = Io * np.exp(-REF_STAIN.T @ C)
    return np.clip(Inorm.T, 0, 255).astype(np.uint8).reshape(h, w, 3)
