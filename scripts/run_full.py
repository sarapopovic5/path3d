#!/usr/bin/env python
"""Run the full path3d pipeline over a manifest.

Every path comes from the environment or the command line, so this script is
safe to commit and identical on every machine. Machine-specific values live in
an untracked env file (see ``scripts/slurm/hpc.env.example``).

In one process, interactively::

    python scripts/run_full.py --manifest m.csv --out run_out --seg-model dpt.pt

or on Slurm as three chained jobs, with the GPU work split over an array::

    bash scripts/slurm/submit_all.sh local/hpc.env

which runs ``--stage register``, then ``--stage sections`` once per array
task, then ``--stage volume``. Later stages take the manifest and section
spacing from what ``register`` recorded in --out.

Omit --seg-model to skip segmentation inference entirely.
"""

# Enabled before any heavy import so a native crash (libvips, OpenSlide, CUDA)
# still prints a Python traceback into the SLURM log instead of dying silently.
import faulthandler

faulthandler.enable()

import argparse
import os
import sys
from pathlib import Path


def _env_default(name: str) -> str | None:
    """Read an env var, treating empty/whitespace as unset."""
    value = os.environ.get(name)
    return value if value and value.strip() else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        default=_env_default("MANIFEST"),
        help="manifest CSV from slide_io.create_manifest. Env: MANIFEST",
    )
    parser.add_argument(
        "--out",
        default=_env_default("OUTDIR"),
        help="output directory for registered sections and results. Env: OUTDIR",
    )
    parser.add_argument(
        "--seg-model",
        default=_env_default("DPT_CKPT"),
        help="DPT decoder checkpoint. Omit to skip segmentation. Env: DPT_CKPT",
    )
    parser.add_argument(
        "--tissue-type",
        default=os.environ.get("TISSUE_TYPE", "HGSC"),
        help="key into config.TISSUE_CLASSES. Env: TISSUE_TYPE (default: HGSC)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("SEG_DEVICE", "cuda"),
        help="torch device for segmentation. Env: SEG_DEVICE (default: cuda)",
    )
    parser.add_argument(
        "--no-nuclei",
        action="store_true",
        help="skip nuclear detection",
    )
    parser.add_argument(
        "--no-non-rigid",
        action="store_true",
        help="skip elastic refinement, rigid registration only",
    )
    parser.add_argument(
        "--segment-first",
        action="store_true",
        help="segment each original slide at the training resolution, then "
        "warp the label map onto the registered canvas, instead of segmenting "
        "the registered image. The register stage records it; the sections "
        "stage follows that record",
    )
    parser.add_argument(
        "--persist-nuclei-masks",
        action="store_true",
        help="write per-section instance masks to zarr (large)",
    )
    parser.add_argument(
        "--voxel-um",
        type=float,
        default=_env_default("VOXEL_UM"),
        help="3D volume and quantification resolution, um/voxel, when "
        "--section-thickness-um/--sampling-interval are not given. "
        "Env: VOXEL_UM (default: config.VOXEL_UM)",
    )
    parser.add_argument(
        "--section-thickness-um",
        type=float,
        default=_env_default("SECTION_THICKNESS_UM"),
        help="thickness each section was cut at, um. Give with "
        "--sampling-interval. Env: SECTION_THICKNESS_UM",
    )
    parser.add_argument(
        "--sampling-interval",
        type=int,
        default=_env_default("SAMPLING_INTERVAL"),
        help="every how many cut sections one was stained and imaged (2 = "
        "every second). Sections are thickness x interval apart, and that is "
        "the voxel size. Env: SAMPLING_INTERVAL",
    )

    parser.add_argument(
        "--stage",
        choices=("all", "register", "sections", "volume"),
        default="all",
        help="run everything in one process (all, the default), or one stage "
        "of a staged run: register (CPU), sections (GPU, this task's share), "
        "volume (CPU, once every task has finished)",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=_env_default("SLURM_ARRAY_TASK_ID") or 0,
        help="sections stage: which task this is, 0-based. "
        "Env: SLURM_ARRAY_TASK_ID (default: 0)",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=_env_default("SLURM_ARRAY_TASK_COUNT") or 1,
        help="sections stage: how many tasks share the sections. "
        "Env: SLURM_ARRAY_TASK_COUNT (default: 1)",
    )

    args = parser.parse_args(argv)

    if (args.section_thickness_um is None) != (args.sampling_interval is None):
        parser.error(
            "give --section-thickness-um and --sampling-interval together "
            "(4 um sections with every second one stained: 4 and 2)"
        )

    required = [("--out/OUTDIR", args.out)]
    if args.stage in ("all", "register"):
        required.append(("--manifest/MANIFEST", args.manifest))
    missing = [flag for flag, value in required if not value]
    if missing:
        parser.error(
            "missing required path(s): "
            + ", ".join(missing)
            + ". Pass the flag, or source your env file first."
        )

    return args


def _print_results(results: list[dict]) -> None:
    # run_pipeline writes label maps and nuclei CSVs per section as they
    # complete; this is only a summary of what landed on disk. 'label_map' is a
    # Path or None, and 'nuclei' is {'n_nuclei', 'path'} or None.
    print(f"\n{len(results)} sections processed", flush=True)
    for result in results:
        nuclei = result.get("nuclei")
        count = nuclei["n_nuclei"] if nuclei else None
        print(
            result["section_index"],
            result.get("registered_path"),
            result.get("label_map"),
            f"n_nuclei={count}" if count is not None else "",
            flush=True,
        )


def _print_report(output_dir: Path) -> None:
    report = output_dir / "quantification" / "report.md"
    if report.exists():
        for volume in sorted(output_dir.glob("volume_*um.zarr")):
            print(f"\nvolume:     {volume}", flush=True)
        print(f"report:     {report}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Fail on a bad path now, with a clear message, rather than partway through
    # a multi-hour job.
    manifest = None
    if args.stage in ("all", "register"):
        manifest = Path(args.manifest).expanduser()
        if not manifest.is_file():
            sys.exit(f"manifest not found: {manifest}")

    seg_model = None
    if args.seg_model and args.stage in ("all", "sections"):
        seg_model = Path(args.seg_model).expanduser()
        if not seg_model.is_file():
            sys.exit(f"segmentation checkpoint not found: {seg_model}")

    output_dir = Path(args.out).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # libvips keeps decoded tiles in a process-wide cache. On whole slide images
    # that grows without bound and is a common cause of OOM kills on a compute
    # node, so it is disabled before the first read.
    import pyvips

    pyvips.cache_set_max(0)
    pyvips.cache_set_max_mem(0)

    import path3d.config as cfg
    import path3d.pipeline as pl
    from path3d.volume import _resolve_voxel_um

    print(f"stage:      {args.stage}", flush=True)
    print(f"output:     {output_dir}", flush=True)
    if args.stage in ("all", "register"):
        print(f"segment:    {'original slide, then warp labels' if args.segment_first else 'registered image'}",
              flush=True)

    if args.stage == "sections":
        print(f"task:       {args.task_index} of {args.n_tasks}", flush=True)
        print(f"seg model:  {seg_model or '(segmentation skipped)'}", flush=True)
        print(f"device:     {args.device}", flush=True)
        try:
            results = pl.run_sections_stage(
                output_dir,
                task_index=args.task_index,
                n_tasks=args.n_tasks,
                seg_model=str(seg_model) if seg_model else None,
                seg_device=args.device,
                detect_nuclei=not args.no_nuclei,
                persist_nuclei_masks=args.persist_nuclei_masks,
            )
        except (FileNotFoundError, ValueError) as exc:
            sys.exit(str(exc))
        _print_results(results)
        return 0

    if args.stage == "volume":
        try:
            pl.run_volume_stage(
                output_dir, voxel_um=args.voxel_um, detect_nuclei=not args.no_nuclei
            )
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            sys.exit(str(exc))
        _print_report(output_dir)
        return 0

    thickness_um, interval = args.section_thickness_um, args.sampling_interval
    try:
        voxel_um, _ = _resolve_voxel_um(
            args.voxel_um if args.voxel_um is not None or thickness_um is not None else cfg.VOXEL_UM,
            thickness_um,
            interval,
        )
    except ValueError as exc:
        sys.exit(str(exc))

    print(f"manifest:   {manifest}", flush=True)
    print(f"voxel:      {voxel_um:g} um", flush=True)
    print(
        "sections:   "
        + (f"{thickness_um:g} um, every {interval} imaged" if thickness_um else "(not given)"),
        flush=True,
    )

    if args.stage == "register":
        try:
            results = pl.run_register_stage(
                str(manifest),
                output_dir,
                tissue_type=args.tissue_type,
                do_non_rigid=not args.no_non_rigid,
                voxel_um=voxel_um,
                section_thickness_um=thickness_um,
                sampling_interval=interval,
                segment_first=args.segment_first,
            )
        except FileExistsError as exc:
            sys.exit(str(exc))
        print(f"\n{len(results)} sections registered", flush=True)
        return 0

    print(f"seg model:  {seg_model or '(segmentation skipped)'}", flush=True)
    print(f"device:     {args.device}", flush=True)

    results = pl.run_pipeline(
        manifest_path=str(manifest),
        output_dir=str(output_dir),
        seg_model=str(seg_model) if seg_model else None,
        tissue_type=args.tissue_type,
        seg_device=args.device,
        detect_nuclei=not args.no_nuclei,
        do_non_rigid=not args.no_non_rigid,
        persist_nuclei_masks=args.persist_nuclei_masks,
        voxel_um=voxel_um,
        section_thickness_um=thickness_um,
        sampling_interval=interval,
        segment_first=args.segment_first,
    )
    _print_results(results)
    _print_report(output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
