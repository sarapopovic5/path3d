"""End-to-end driver: registered sections -> per-section niches -> 3D volume.

The niche model card is explicit that it does not solve the alignment problem::

    Tile coordinates are in each slide's own level-0 pixels, so sections are
    not co-registered to one another -- align them yourself.

path3d is that alignment step, so this runs **after** ``pipeline.run_pipeline``
and consumes the registered OME-TIFFs it wrote to ``output_dir/registered/``.
Those share one aligned canvas, which is what makes the per-section grids
stackable.

Predicting on the registered sections rather than the raw slides also sidesteps
a second problem: ``predict_niches.py`` reads through ``tifffile``/``zarr`` and
cannot open a CZI at all, while ``registration.warp_and_save_section`` has a
purpose-built CZI adapter (see ``registration._CziReaderAdapter``) that avoids
the Bio-Formats JPEGXR segfault on this dataset.

Resolution matters here. The model takes a 112 um window and resizes it to
224x224, so at the default ``cfg.SEG_MPP`` of 0.5 um/px that window is already
224 px and carries visibly less detail than the 509 px UNI2-h saw in training.
Warp at the slide's native mpp (~0.22) for the niche pass; the output is
pyramidal, so ``segmentation.predict_section`` can still pick a ~0.44 um/px
level off the same file and you only pay for one warp.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from path3d.niches.predict import (
    NicheModel,
    canvas_wh,
    load_encoder,
    load_model,
    predict_slide,
    read_mpp,
)
from path3d.niches.rasterize import NicheSection, rasterize_tiles
from path3d.niches.volume import build_niche_volume
from path3d.slide_io import load_manifest


def registered_path_for(
    registered_dir: str | Path, section_index: int
) -> Path:
    """Where ``pipeline.run_pipeline`` wrote section ``section_index``."""
    return Path(registered_dir) / f"{section_index:04d}.ome.tiff"


def predict_sections(
    manifest_csv: str | Path,
    registered_dir: str | Path,
    output_dir: str | Path,
    *,
    model: NicheModel | str | Path | None = None,
    mpp: float | None = None,
    quantile: float = 0.80,
    mask_level: int = 4,
    limit: int = 0,
    stride: int = 1,
    device: str | None = None,
    skip_existing: bool = True,
    cache_embeddings: bool = True,
    verbose: bool = True,
) -> dict[str, dict]:
    """Run the niche model over every registered section in a manifest.

    Writes ``output_dir/tiles/{idx:04d}_tiles_niches.csv`` and
    ``output_dir/tiles/{idx:04d}_meta.json`` per section as it goes, so a crash
    or walltime kill part-way through a long stack loses only the section in
    flight. The ~2.5 GB UNI2-h encoder is loaded once and reused.

    Args:
        manifest_csv: manifest CSV; row order is z order.
        registered_dir: directory of registered OME-TIFFs
            (``pipeline.run_pipeline``'s ``output_dir/registered``).
        output_dir: where per-section CSVs and metadata are written.
        model: :class:`NicheModel`, bundle path, or None for the packaged one.
        mpp: level-0 microns per pixel of the REGISTERED files; read from each
            file's own metadata when None.
        quantile: per-slide quantile for the ``call_*`` columns.
        mask_level: pyramid level for tissue detection.
        limit: cap tiles per section (debugging only).
        stride: keep every ``stride``-th grid cell in each axis -- a fast
            full-extent preview of the whole stack at 1/stride^2 the cost.
        device: torch device override.
        skip_existing: reuse a section's CSV if it is already on disk.
        cache_embeddings: checkpoint each section's UNI2-h embeddings to
            ``output_dir/embeddings/{idx:04d}.npz``. Costs ~75 MB per 25 000
            tiles and makes an interrupted section resume mid-way rather than
            restart; it also lets the classifier be re-run later without
            paying for the encoder again.
        verbose: log progress.

    Returns:
        ``str(manifest path)`` -> metadata dict, for sections that produced
        tiles. Sections whose registered file is missing are skipped with a
        warning rather than aborting the run.
    """
    paths = load_manifest(manifest_csv)
    tiles_dir = Path(output_dir) / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = Path(output_dir) / "embeddings"
    if cache_embeddings:
        embeddings_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(model, NicheModel):
        model = load_model(model)

    encoder = None
    metas: dict[str, dict] = {}

    for index, path in enumerate(paths):
        registered = registered_path_for(registered_dir, index)
        csv_path = tiles_dir / f"{index:04d}_tiles_niches.csv"
        meta_path = tiles_dir / f"{index:04d}_meta.json"

        if skip_existing and csv_path.exists() and meta_path.exists():
            metas[str(path)] = json.loads(meta_path.read_text())
            if verbose:
                print(f"[{index:04d}] cached -> {csv_path.name}", flush=True)
            continue

        if not registered.exists():
            print(
                f"[WARN] [{index:04d}] no registered file at {registered}; "
                f"skipping. Registration may have failed for this section.",
                flush=True,
            )
            continue

        if verbose:
            print(f"\n[{index:04d}] {registered.name}", flush=True)

        if encoder is None:
            encoder = load_encoder(device)

        try:
            tiles, meta = predict_slide(
                registered,
                mpp=mpp,
                model=model,
                quantile=quantile,
                mask_level=mask_level,
                limit=limit,
                stride=stride,
                encoder=encoder,
                cache_path=(
                    embeddings_dir / f"{index:04d}.npz" if cache_embeddings else None
                ),
                verbose=verbose,
            )
        except ValueError as exc:
            print(f"[WARN] [{index:04d}] {exc}; skipping.", flush=True)
            continue

        meta["section_index"] = index
        meta["manifest_path"] = str(path)
        meta["registered_path"] = str(registered)

        tiles.to_csv(csv_path, index=False)
        meta_path.write_text(json.dumps(meta, indent=2))
        metas[str(path)] = meta

    return metas


def load_sections(
    manifest_csv: str | Path,
    output_dir: str | Path,
    *,
    grid_um: float = 32.0,
    registered_dir: str | Path | None = None,
) -> dict[str, NicheSection]:
    """Rasterise the per-section CSVs written by :func:`predict_sections`.

    Every section is rasterised on the canvas recorded in its own metadata, so
    the grids come out identically shaped as long as the sections really were
    warped onto one aligned canvas. ``build_niche_volume`` re-checks that and
    fails loudly if not.

    Args:
        manifest_csv: the same manifest used for prediction.
        output_dir: :func:`predict_sections`' output directory.
        grid_um: rasterisation grid, microns. 32.0 is the model's native tile
            size; pass e.g. 250.0 to reproduce the model card's reporting
            resolution.
        registered_dir: fall back to reading canvas/mpp from the registered
            OME-TIFFs when a section's ``_meta.json`` is absent.

    Returns:
        ``str(manifest path)`` -> :class:`NicheSection`, for sections that have
        a CSV on disk.
    """
    paths = load_manifest(manifest_csv)
    tiles_dir = Path(output_dir) / "tiles"
    sections: dict[str, NicheSection] = {}

    for index, path in enumerate(paths):
        csv_path = tiles_dir / f"{index:04d}_tiles_niches.csv"
        meta_path = tiles_dir / f"{index:04d}_meta.json"
        if not csv_path.exists():
            continue

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            canvas = tuple(meta["canvas_wh"])
            mpp = float(meta["mpp"])
            classes = list(meta["classes"])
        elif registered_dir is not None:
            registered = registered_path_for(registered_dir, index)
            canvas = canvas_wh(registered)
            mpp = read_mpp(registered)
            classes = load_model().classes
        else:
            raise FileNotFoundError(
                f"{meta_path} is missing and registered_dir was not given, so "
                f"the canvas size and mpp for section {index} are unknown."
            )

        sections[str(path)] = rasterize_tiles(
            pd.read_csv(csv_path),
            canvas,  # type: ignore[arg-type]
            mpp,
            classes,
            grid_um=grid_um,
        )

    return sections


def run_niche_pipeline(
    manifest_csv: str | Path,
    registered_dir: str | Path,
    output_dir: str | Path,
    *,
    z_spacing_um: float | None = None,
    grid_um: float = 32.0,
    smooth_um: float = 0.0,
    target_voxel_um: float | None = None,
    model: NicheModel | str | Path | None = None,
    mpp: float | None = None,
    quantile: float = 0.80,
    mask_level: int = 4,
    device: str | None = None,
    write_zarr: bool = True,
    verbose: bool = True,
):
    """Predict niches for every registered section and assemble the 3D volume.

    Args:
        manifest_csv: manifest CSV; row order is z order.
        registered_dir: ``pipeline.run_pipeline``'s ``output_dir/registered``.
        output_dir: destination for per-section CSVs and the volume.
        z_spacing_um: microns between consecutive manifest rows; read from a
            uniform manifest ``thickness_um`` when None.
        grid_um: rasterisation grid, microns.
        smooth_um: in-plane Gaussian sigma in microns applied to probabilities
            before the argmax.
        target_voxel_um: force isotropic voxels (needed by
            ``quantification.compute_volumetrics``).
        model, mpp, quantile, mask_level, device: passed to
            :func:`predict_sections`.
        write_zarr: write the volume to ``output_dir/niche_volume.zarr``.
        verbose: log progress.

    Returns:
        The assembled ``SpatialData``.
    """
    output_dir = Path(output_dir)
    predict_sections(
        manifest_csv,
        registered_dir,
        output_dir,
        model=model,
        mpp=mpp,
        quantile=quantile,
        mask_level=mask_level,
        device=device,
        verbose=verbose,
    )
    sections = load_sections(
        manifest_csv, output_dir, grid_um=grid_um, registered_dir=registered_dir
    )
    if not sections:
        raise RuntimeError(
            f"No per-section niche tables found under {output_dir / 'tiles'}. "
            f"Did prediction run, and does {registered_dir} contain the "
            f"registered OME-TIFFs?"
        )

    return build_niche_volume(
        manifest_csv,
        sections,
        z_spacing_um=z_spacing_um,
        smooth_um=smooth_um,
        target_voxel_um=target_voxel_um,
        output_path=(output_dir / "niche_volume.zarr") if write_zarr else None,
    )
