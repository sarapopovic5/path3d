#!/usr/bin/env python
"""Run the full path3d pipeline over a manifest.

Every path comes from the environment or the command line, so this script is
safe to commit and identical on every machine. Machine-specific values live in
an untracked env file (see ``local/hpc.env``) that is sourced before submission:

    source local/hpc.env && sbatch scripts/slurm/run_full.sbatch

or, interactively::

    python scripts/run_full.py --manifest m.csv --out run_out --seg-model dpt.pt

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

    args = parser.parse_args(argv)

    if (args.section_thickness_um is None) != (args.sampling_interval is None):
        parser.error(
            "give --section-thickness-um and --sampling-interval together "
            "(4 um sections with every second one stained: 4 and 2)"
        )

    missing = [
        flag
        for flag, value in (("--manifest/MANIFEST", args.manifest), ("--out/OUTDIR", args.out))
        if not value
    ]
    if missing:
        parser.error(
            "missing required path(s): "
            + ", ".join(missing)
            + ". Pass the flag, or source your env file first."
        )

    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Fail on a bad path now, with a clear message, rather than partway through
    # a multi-hour job.
    manifest = Path(args.manifest).expanduser()
    if not manifest.is_file():
        sys.exit(f"manifest not found: {manifest}")

    seg_model = None
    if args.seg_model:
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
    from path3d.pipeline import run_pipeline
    from path3d.volume import _resolve_voxel_um

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
    print(f"output:     {output_dir}", flush=True)
    print(f"seg model:  {seg_model or '(segmentation skipped)'}", flush=True)
    print(f"device:     {args.device}", flush=True)
    print(f"voxel:      {voxel_um:g} um", flush=True)
    print(
        "sections:   "
        + (f"{thickness_um:g} um, every {interval} imaged" if thickness_um else "(not given)"),
        flush=True,
    )

    results = run_pipeline(
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
    )

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

    report = output_dir / "quantification" / "report.md"
    if report.exists():
        print(f"\nvolume:     {output_dir / f'volume_{voxel_um:g}um.zarr'}", flush=True)
        print(f"report:     {report}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
