"""Build the whole-stack niche volume as one run directory.

Mirrors the layout of a ``pipeline.run_pipeline`` + ``rebuild_volume`` run
(e.g. ``full_volume_dpt/``), so the result is viewed the same way::

    OUT/
      manifest_thickness_8um.csv    manifest + the z spacing actually used
      volume_build_metadata.json    every parameter, plus per-section status
      label_maps/NNNN_labels.png    per-section argmax, 32 um grid, uint8
      label_maps/NNNN_labels_rgb.png   same, in the HGSC_niches palette
      quantification/soft_volumetrics.csv    per-class volume from probabilities
      quantification/per_section_profile.csv per-section area, for QC
      volume_niches_32um.zarr       labels["tissue_labels"] + images["niche_probabilities"]
                                    (+ tables["nuclei"] with --nuclei-parquet)
      niches/tiles/                 per-section tile CSVs (the model output)
      niches/embeddings/            UNI2-h checkpoints (resume + free re-scoring)

Prediction is the only expensive step and is resumable per section, so on a
cluster just resubmit the same command after a walltime kill. Everything after
it runs in about a minute from ``niches/tiles`` alone -- pass
``--skip-predict`` to rebuild the volume (e.g. with different smoothing)
without a GPU.

    python -m path3d.niches.run \\
        --manifest manifest_hpc.csv --registered RUN/registered --out OUT \\
        --z-spacing-um 8 --voxel-um 32 \\
        --nuclei-parquet full_volume_dpt/nuclei_dpt_8um.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from path3d.config import TISSUE_CLASS_COLORS
from path3d.niches.pipeline import load_sections, predict_sections
from path3d.niches.predict import load_model
from path3d.niches.quantify import per_section_profile, soft_volumetrics
from path3d.niches.rasterize import NicheSection
from path3d.niches.volume import build_niche_volume
from path3d.slide_io import load_manifest

TISSUE_TYPE = "HGSC_niches"

# The niche model was trained at this level-0 resolution. Coarser registered
# sections still produce the right PHYSICAL window (112 um), just with less
# detail in it than the encoder saw in training.
_TRAINING_MPP = 0.2201


def write_label_maps(labels: np.ndarray, out_dir: Path) -> None:
    """Write one class-index PNG and one RGB PNG per z-slice.

    Same file names and palette convention as
    ``pipeline._write_section_outputs``, which is not reused directly only
    because importing ``path3d.pipeline`` pulls in VALIS and cellpose.
    """
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    palette = TISSUE_CLASS_COLORS[TISSUE_TYPE]
    for k, plane in enumerate(labels):
        plane = plane.astype(np.uint8)
        Image.fromarray(plane).save(out_dir / f"{k:04d}_labels.png")
        rgb = palette[np.clip(plane, 0, len(palette) - 1)]
        Image.fromarray(rgb).save(out_dir / f"{k:04d}_labels_rgb.png")


def _empty_like(section: NicheSection) -> NicheSection:
    """A section with no predictions, on the same grid as ``section``."""
    return NicheSection(
        probs=np.full_like(section.probs, np.nan),
        classes=list(section.classes),
        tile_count=np.zeros_like(section.tile_count),
        grid_um=section.grid_um,
        canvas_wh=section.canvas_wh,
        mpp=section.mpp,
    )


def build_niche_run(
    manifest_csv: str | Path,
    registered_dir: str | Path,
    out_dir: str | Path,
    *,
    z_spacing_um: float,
    voxel_um: float = 32.0,
    grid_um: float = 32.0,
    smooth_um: float = 0.0,
    smooth_z_sigma: float = 0.0,
    quantile: float = 0.80,
    mask_level: int = 4,
    stride: int = 1,
    device: str | None = None,
    cache_embeddings: bool = True,
    nuclei_parquet: str | Path | None = None,
    allow_missing: bool = False,
    predict: bool = True,
    tag: str = "niches",
) -> Path:
    """Predict every section, then write the full run directory.

    Args:
        manifest_csv: manifest; row order is z order.
        registered_dir: the registered ``NNNN.ome.tiff`` sections.
        out_dir: run directory to write (see the module docstring).
        z_spacing_um: microns between consecutive manifest rows.
        voxel_um: cube size of the saved volume. Must be a whole multiple of
            ``z_spacing_um``; that many sections are averaged per voxel.
        grid_um: in-plane rasterisation grid, microns.
        smooth_um, smooth_z_sigma: see :func:`build_niche_volume`.
        quantile, mask_level, stride, device, cache_embeddings: see
            :func:`predict_sections`.
        nuclei_parquet: a merged nuclear table written by
            ``volume.build_volume`` (e.g. ``nuclei_dpt_8um.parquet``), attached
            as ``tables["nuclei"]``. Nuclei from sections left out of the cube
            are dropped so the table matches the volume.
        allow_missing: build even if some sections have no prediction, leaving
            them empty. Off by default: a missing section is usually a job
            that has not finished, not a blank slide.
        predict: False skips prediction and builds from ``niches/tiles`` only.
        tag: volume name, ``volume_{tag}_{voxel_um}um.zarr``.

    Returns:
        Path of the written zarr.

    Raises:
        RuntimeError: sections have no prediction and ``allow_missing`` is
            False.
    """
    manifest_csv = Path(manifest_csv)
    registered_dir = Path(registered_dir)
    out_dir = Path(out_dir)
    niches_dir = out_dir / "niches"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model()
    if predict:
        predict_sections(
            manifest_csv,
            registered_dir,
            niches_dir,
            model=model,
            quantile=quantile,
            mask_level=mask_level,
            stride=stride,
            device=device,
            cache_embeddings=cache_embeddings,
        )

    paths = load_manifest(manifest_csv)
    sections = load_sections(
        manifest_csv, niches_dir, grid_um=grid_um, registered_dir=registered_dir
    )
    missing = [i for i, p in enumerate(paths) if str(p) not in sections]
    if len(missing) == len(paths):
        raise RuntimeError(f"No section has predictions under {niches_dir / 'tiles'}.")
    if missing:
        if not allow_missing:
            raise RuntimeError(
                f"{len(missing)} of {len(paths)} sections have no prediction: "
                f"{missing}. If the prediction job was cut off, resubmit it -- "
                f"finished sections are skipped. If these sections really are "
                f"blank, pass --allow-missing to leave them empty."
            )
        print(f"[WARN] leaving {len(missing)} section(s) empty: {missing}", flush=True)
        reference = next(iter(sections.values()))
        for i in missing:
            sections[str(paths[i])] = _empty_like(reference)

    mpps = sorted({round(s.mpp, 6) for s in sections.values()})
    if max(mpps) > 1.2 * _TRAINING_MPP:
        print(
            f"[WARN] registered sections are {max(mpps):g} um/px; the niche "
            f"model was trained at {_TRAINING_MPP} um/px, so each 112 um window "
            f"carries ~{(max(mpps) / _TRAINING_MPP) ** 2:.1f}x fewer pixels than "
            f"in training. See docs/niches/INTEGRATION.md.",
            flush=True,
        )

    # Native (z_spacing, grid, grid) volume: one slice per section, for the
    # per-section label maps and QC profile. Not written.
    native = build_niche_volume(
        manifest_csv,
        sections,
        z_spacing_um=z_spacing_um,
        smooth_um=smooth_um,
        smooth_z_sigma=smooth_z_sigma,
    )
    write_label_maps(np.asarray(native.labels["tissue_labels"].data), out_dir / "label_maps")
    quant_dir = out_dir / "quantification"
    quant_dir.mkdir(parents=True, exist_ok=True)
    per_section_profile(native).to_csv(quant_dir / "per_section_profile.csv", index=False)
    del native

    cube = build_niche_volume(
        manifest_csv,
        sections,
        z_spacing_um=z_spacing_um,
        smooth_um=smooth_um,
        smooth_z_sigma=smooth_z_sigma,
        target_voxel_um=voxel_um,
    )
    dropped = list(cube.attrs["niche_dropped_sections"])
    volumetrics = soft_volumetrics(cube, csv_path=quant_dir / "soft_volumetrics.csv")

    n_nuclei = None
    if nuclei_parquet is not None:
        from path3d.volume import nuclei_table

        nuclei = pd.read_parquet(nuclei_parquet)
        if dropped and "section_index" in nuclei.columns:
            nuclei = nuclei[~nuclei["section_index"].isin(dropped)]
        cube.tables["nuclei"] = nuclei_table(nuclei)
        n_nuclei = int(len(nuclei))

    zarr_path = out_dir / f"volume_{tag}_{voxel_um:g}um.zarr"
    cube.write(str(zarr_path), overwrite=True)

    manifest = pd.read_csv(manifest_csv)
    manifest["thickness_um"] = z_spacing_um
    manifest_out = out_dir / f"manifest_thickness_{z_spacing_um:g}um.csv"
    manifest.to_csv(manifest_out, index=False)

    tiles_dir = niches_dir / "tiles"
    section_rows = []
    for i, p in enumerate(paths):
        meta_path = tiles_dir / f"{i:04d}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        section_rows.append(
            {
                "section_index": i,
                "path": str(p),
                "has_prediction": i not in missing,
                "in_volume": i not in dropped,
                "n_tiles": meta.get("n_tiles"),
                "tissue_frac": meta.get("tissue_frac"),
            }
        )

    metadata = {
        "kind": "niche_volume",
        "manifest_csv": str(manifest_csv),
        "registered_dir": str(registered_dir),
        "tissue_type": TISSUE_TYPE,
        "classes": list(model.classes),
        "model": {
            "scheme": model.scheme,
            "trained_on": model.trained_on,
            "loso_auc_by_class": model.auc_by_class,
            "tile_um": model.tile_um,
            "fov_um": model.fov_um,
        },
        "registered_mpp": mpps,
        "grid_um": grid_um,
        "z_spacing_um": z_spacing_um,
        "voxel_um": voxel_um,
        "sections_per_voxel": int(cube.attrs["niche_sections_per_voxel"]),
        "dropped_sections": dropped,
        "missing_sections": missing,
        "smooth_um": smooth_um,
        "smooth_z_sigma": smooth_z_sigma,
        "quantile": quantile,
        "stride": stride,
        "volume": zarr_path.name,
        "volume_shape_zyx": [int(n) for n in cube.labels["tissue_labels"].shape],
        "nuclei_parquet": str(nuclei_parquet) if nuclei_parquet else None,
        "n_nuclei": n_nuclei,
        "label_maps_dir": "label_maps",
        "niches_dir": "niches",
        "n_sections": len(paths),
        "sections": section_rows,
    }
    (out_dir / "volume_build_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\n[VOLUME] wrote {zarr_path}  shape {metadata['volume_shape_zyx']} "
          f"at {voxel_um:g} um cubes", flush=True)
    print(volumetrics[["class_name", "expected_volume_mm3", "volume_fraction"]]
          .to_string(index=False), flush=True)
    return zarr_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m path3d.niches.run",
        description="Predict niches for every registered section and write "
        "the whole-stack niche volume run directory.",
    )
    ap.add_argument("--manifest", required=True, help="manifest CSV (row order = z order)")
    ap.add_argument("--registered", required=True, help="directory of NNNN.ome.tiff")
    ap.add_argument("--out", required=True, help="run directory to write")
    ap.add_argument("--z-spacing-um", type=float, required=True,
                    help="microns between consecutive manifest rows")
    ap.add_argument("--voxel-um", type=float, default=32.0,
                    help="cube size; a whole multiple of --z-spacing-um (default 32)")
    ap.add_argument("--grid-um", type=float, default=32.0,
                    help="in-plane grid (default 32, the model's tile size)")
    ap.add_argument("--smooth-um", type=float, default=0.0,
                    help="in-plane Gaussian sigma in microns (default 0, off)")
    ap.add_argument("--smooth-z-sigma", type=float, default=0.0,
                    help="z Gaussian sigma in sections (default 0, off)")
    ap.add_argument("--quantile", type=float, default=0.80)
    ap.add_argument("--mask-level", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1,
                    help="preview: every Nth tile per axis. Use a separate --out.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-cache", action="store_true", help="do not checkpoint embeddings")
    ap.add_argument("--nuclei-parquet", default=None,
                    help="merged nuclear table to attach, e.g. nuclei_dpt_8um.parquet")
    ap.add_argument("--allow-missing", action="store_true",
                    help="build even if some sections have no prediction")
    ap.add_argument("--skip-predict", action="store_true",
                    help="build from existing niches/tiles only (no GPU)")
    ap.add_argument("--tag", default="niches", help="volume_{tag}_{voxel}um.zarr")
    a = ap.parse_args(argv)

    build_niche_run(
        a.manifest,
        a.registered,
        a.out,
        z_spacing_um=a.z_spacing_um,
        voxel_um=a.voxel_um,
        grid_um=a.grid_um,
        smooth_um=a.smooth_um,
        smooth_z_sigma=a.smooth_z_sigma,
        quantile=a.quantile,
        mask_level=a.mask_level,
        stride=a.stride,
        device=a.device,
        cache_embeddings=not a.no_cache,
        nuclei_parquet=a.nuclei_parquet,
        allow_missing=a.allow_missing,
        predict=not a.skip_predict,
        tag=a.tag,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
