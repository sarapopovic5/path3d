"""Xenium-derived tissue neighbourhoods from H&E, in 3D.

Predicts a four-class neighbourhood map (``epithelium | immune | stroma |
acellular``) per 32 um tile from H&E alone, then stacks the per-section maps
into a 3D volume using path3d's registration.

The classifier and its stain normaliser are vendored from the Cook Lab
``2026_final_xenium_analysis`` package; ``docs/niches/MODEL_CARD.md`` is their
model card, unmodified, and is the authority on validation and limits. Read it
before trusting a number -- in particular:

* **epithelium and immune are the dependable classes** (LOTO AUC 0.911 /
  0.923). Stroma and acellular are regional tendencies, not firm calls.
* **HGSC only**, post-Xenium H&E, one scanner. Anything else is extrapolation.
* Probabilities transfer, a global argmax does not. Prevalence ranged 4.5-98.4%
  across their eight tumours.

Typical use, after ``pipeline.run_pipeline`` has written registered sections::

    from path3d.niches import run_niche_pipeline

    sdata = run_niche_pipeline(
        "manifest.csv",
        "run_out/registered",
        "run_out/niches",
        z_spacing_um=12.0,   # spacing between CONSECUTIVE MANIFEST ROWS
        smooth_um=100.0,
    )

See ``docs/niches/INTEGRATION.md`` for the resolution, z-pitch and
argmax-bias decisions this wraps.
"""

from path3d.niches.pipeline import (
    load_sections,
    predict_sections,
    registered_path_for,
    run_niche_pipeline,
)
from path3d.niches.predict import (
    NicheModel,
    canvas_wh,
    classify_embeddings,
    embed_tiles,
    grid_shape,
    load_embedding_cache,
    load_encoder,
    load_model,
    predict_slide,
    read_mpp,
    tile_fingerprint,
    tile_grid,
    tissue_mask,
)
from path3d.niches.rasterize import (
    NicheSection,
    argmax_labels,
    rasterize_csv,
    rasterize_tiles,
    smooth_probs,
)
from path3d.niches.volume import build_niche_volume, resolve_z_spacing

__all__ = [
    "NicheModel",
    "NicheSection",
    "argmax_labels",
    "build_niche_volume",
    "canvas_wh",
    "classify_embeddings",
    "embed_tiles",
    "grid_shape",
    "load_embedding_cache",
    "load_encoder",
    "load_model",
    "load_sections",
    "predict_sections",
    "predict_slide",
    "rasterize_csv",
    "rasterize_tiles",
    "read_mpp",
    "registered_path_for",
    "resolve_z_spacing",
    "run_niche_pipeline",
    "smooth_probs",
    "tile_fingerprint",
    "tile_grid",
    "tissue_mask",
]
