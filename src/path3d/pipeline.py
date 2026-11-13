from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

import path3d.slide_io as io
import path3d.preprocessing as pr
import path3d.registration as reg
import path3d.segmentation as sg
import path3d.nuclear_detection as nd
import path3d.volume as vol
import path3d.quantification as qt
import path3d.config as cfg


# resolution used only for tissue mask computation. predict_section later
# upsamples this to the seg_mpp = 0.5 label grid with order=0 nearest-
# neighbour. 2.0 um/px is a DELIBERATE ALGORITHMIC DIVERGENCE from
# segmentation_helpers/inference*.py's standalone 8.0 um/px visualization-
# resolution default, not an oversight: at 8 um/px, make_tissue_mask's
# low_saturation < 0.08 check flags pale/lightly-eosin-stained stroma as
# background, because coarser sampling further dilutes stroma's already-
# weak saturation. predict_section then hard-forces any tile with < 0.1
# tissue coverage to class 0 ("space") before the model ever sees it, so
# the stroma is silently lost. Confirmed empirically on a live HPC run and
# reproduced independently by running the standalone inference script
# against the same registered images.
_MASK_MPP = 2.0

# segment_first tolerance: a source pyramid level counts as the training
# resolution when it is at most this much coarser than cfg.SEG_TRAIN_MPP
# (CZI level 1 is 0.44035 um/px against the 0.4402 the checkpoint saw).
_TRAIN_LEVEL_TOL = 0.01


# DATA FLOW: one slide is open at a time so as to not overload RAM


def run_section(slide: io.SlideReader, section_index: int) -> dict:
    """Process one section through its per-slide stages (tissue mask + QC prep).

    QC pass/fail itself is decided later, by pr.flag_qc_outliers on the
    full stack of results.

    Args:
        slide: Open SlideReader for this section (caller owns it).
        section_index: 0-based z-index of this section (from manifest).

    Returns:
        Per-section results dict, including ``level0_mpp`` (native µm/px) and
        ``level0_shape_rc`` so later stages dont need to reopen the slide
    """
    results: dict = {"section_index": section_index}

    # tissue mask + QC (80 mpp — coarsest, cheapest)
    img_80 = pr.downsample_to_mpp(slide, 80.0)
    results["tissue_mask_80"] = pr.make_tissue_mask(img_80)
    results["img_80"] = img_80
    results["level0_mpp"] = slide.get_mpp()
    level0_w, level0_h = slide.level_dimensions[0]
    results["level0_shape_rc"] = (int(level0_h), int(level0_w))

    return results


def _training_level_mpp(slide: io.SlideReader, train_mpp: float = cfg.SEG_TRAIN_MPP) -> float:
    """Resolution (µm/px) of the coarsest pyramid level of ``slide`` that is
    still no coarser than ``train_mpp`` (1% tolerance), so segmenting there
    shows the model tissue at the scale it was trained on without upsampling.

    For the pilot CZIs (0.22017 µm/px, 2x levels) this is level 1, 0.44035
    µm/px: the exact level the training patches were read from.
    """
    mpp0 = slide.get_mpp()
    limit = train_mpp * (1 + _TRAIN_LEVEL_TOL)
    best = 0
    for level, ds in enumerate(slide.level_downsamples):
        if mpp0 * ds <= limit:
            best = level
    return mpp0 * slide.level_downsamples[best]


def _save_label_png(label_map: np.ndarray, path: Path) -> Path:
    """Write a uint8 label PNG under a temporary name, then rename it, so a
    killed job never leaves a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    Image.fromarray(label_map.astype(np.uint8)).save(tmp_path, format="PNG")
    tmp_path.replace(path)
    return path


def _segment_raw_section(
    path: Path,
    seg_model_obj,
    *,
    num_classes: int,
    section_idx: int,
    seg_device: str,
) -> tuple[np.ndarray, float]:
    """Segment one ORIGINAL (unregistered) slide at the training resolution.

    The tissue mask comes from the same slide, so it never sees the canvas
    fill a registered image has around the section.

    Returns:
        ((H, W) uint8 label map covering the whole slide, its µm/px).
    """
    with io.open_slide(path) as slide:
        raw_mpp = _training_level_mpp(slide)
        tissue_mask = pr.make_tissue_mask(pr.downsample_to_mpp(slide, _MASK_MPP))
    label_map = sg.predict_section(
        path,
        seg_model_obj,
        tissue_mask,
        num_classes=num_classes,
        section_idx=section_idx,
        seg_mpp=raw_mpp,
        device=seg_device,
    )
    return label_map.astype(np.uint8), raw_mpp


def _write_section_outputs(
    result: dict,
    output_dir: Path,
    *,
    tissue_type: str,
) -> dict:
    """Persist whatever a section produced so a mid-run crash doesn't lose it.

    Writes (skips missing keys, never raises):
        output_dir/label_maps/{idx:04d}_labels.png     -- label map, uint8.
        output_dir/label_maps/{idx:04d}_labels_rgb.png -- same map colored via
            cfg.TISSUE_CLASS_COLORS[tissue_type], for visual inspection.
        output_dir/nuclei/{idx:04d}_nuclei.csv -- nuclei feature table.

    Args:
        result: Per-section result dict from run_pipeline's second loop.
        output_dir: Root output directory (same one passed to run_pipeline).
        tissue_type: Key into cfg.TISSUE_CLASS_COLORS for the RGB palette.

    Returns:
        Dict of what was actually written this call, so callers (see
        ``_release_section_payload``) can swap in-memory values for on-disk
        paths without re-deriving filenames:
            {"label_map_path": Path | None, "nuclei_csv_path": Path | None}.
    """
    idx = result["section_index"]
    written: dict = {"label_map_path": None, "nuclei_csv_path": None}

    label_map = result.get("label_map")
    if label_map is not None:
        label_map_dir = output_dir / "label_maps"
        label_map_dir.mkdir(parents=True, exist_ok=True)

        label_map_u8 = label_map.astype(np.uint8)
        label_map_path = _save_label_png(label_map_u8, label_map_dir / f"{idx:04d}_labels.png")
        written["label_map_path"] = label_map_path

        palette = cfg.TISSUE_CLASS_COLORS[tissue_type]
        clamped = np.clip(label_map_u8, 0, len(palette) - 1)
        Image.fromarray(palette[clamped]).save(
            label_map_dir / f"{idx:04d}_labels_rgb.png"
        )

    nuclei = result.get("nuclei")
    if nuclei is not None:
        nuclei_dir = output_dir / "nuclei"
        nuclei_dir.mkdir(parents=True, exist_ok=True)

        df = nuclei["features"] if isinstance(nuclei, dict) else nuclei
        nuclei_csv_path = nuclei_dir / f"{idx:04d}_nuclei.csv"
        # Written under a temporary name and renamed, so a job killed mid-write
        # never leaves a truncated table that _section_done would count as done.
        tmp_path = nuclei_csv_path.with_name(nuclei_csv_path.name + ".tmp")
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(nuclei_csv_path)
        written["nuclei_csv_path"] = nuclei_csv_path

    return written


def _section_done(output_dir: Path, idx: int, *, segment: bool, detect_nuclei: bool) -> bool:
    """Whether a section's outputs from an earlier attempt are already on disk.

    The nuclei CSV is the last file ``_write_section_outputs`` writes, and it
    is renamed into place, so its presence means the section finished.
    """
    if not (segment or detect_nuclei):
        return False
    if segment and not (output_dir / "label_maps" / f"{idx:04d}_labels.png").exists():
        return False
    if detect_nuclei and not (output_dir / "nuclei" / f"{idx:04d}_nuclei.csv").exists():
        return False
    return True


def task_sections(n_sections: int, task_index: int, n_tasks: int) -> range:
    """Manifest rows one task of an ``n_tasks`` array processes.

    Interleaved (task 0 gets rows 0, n_tasks, 2*n_tasks, ...) rather than in
    contiguous blocks, so every task gets sections from the whole depth of the
    block and the tasks finish at about the same time.
    """
    if n_tasks < 1 or not 0 <= task_index < n_tasks:
        raise ValueError(
            f"task_index must be in [0, n_tasks), got {task_index} of {n_tasks}"
        )
    return range(task_index, n_sections, n_tasks)


def _add_registered_coordinates(
    nuclei,
    valis_slide,
    *,
    level0_shape_rc: tuple[int, int],
    reg_mpp: float,
    target_mpp: float,
    non_rigid: bool,
) -> None:
    """Add ``reg_y_um``/``reg_x_um`` (registered canvas, µm) to a section's
    nuclei table in place. ``nuclei`` is what ``detect_nuclei_section``
    returned: a DataFrame, or a dict holding it under ``"features"``."""
    df = nuclei["features"] if isinstance(nuclei, dict) else nuclei
    reg_yx = reg.warp_points_to_registered(
        valis_slide,
        df[["centroid_y", "centroid_x"]].to_numpy(),
        level0_shape_rc=level0_shape_rc,
        reg_mpp=reg_mpp,
        target_mpp=target_mpp,
        non_rigid=non_rigid,
    )
    df["reg_y_um"] = reg_yx[:, 0]
    df["reg_x_um"] = reg_yx[:, 1]


def _release_section_payload(result: dict, written: dict) -> None:
    """Swap a section's heavy in-memory values for on-disk references.

    Keeps ``run_pipeline``'s retained memory bounded to one section's
    payload at a time, instead of accumulating every section's full label
    map and nuclei table in ``all_results`` for the whole run (OOM-killed a real 150+ section HPC run)

    Mutates ``result`` in place:
        result["label_map"] -- the on-disk label-PNG Path, or None.
        result["nuclei"]    -- {"n_nuclei": int, "path": Path}, or None.
            (mask_path is not carried through -- it's derivable as
            output_dir/nuclei/{idx:04d}.zarr when needed.)

    Args:
        result: Per-section result dict, mutated in place.
        written: Return value of ``_write_section_outputs`` for this result.
    """
    result["label_map"] = written["label_map_path"]

    nuclei = result.get("nuclei")
    if nuclei is None:
        result["nuclei"] = None
    else:
        df = nuclei["features"] if isinstance(nuclei, dict) else nuclei
        result["nuclei"] = {
            "n_nuclei": int(len(df)),
            "path": written["nuclei_csv_path"],
        }


def _write_volume_build_inputs(
    output_dir: Path,
    slide_paths: list[Path],
    transforms: dict[int, object],
    *,
    manifest_path: str | Path,
    seg_mpp: float,
    reg_mpp: float,
    level0_mpp: float | None,
    section_thickness_um: float | None,
    sampling_interval: int | None,
    do_non_rigid: bool,
    tissue_type: str,
    segment_first: bool = False,
) -> None:
    """Persist registration transforms + volume-build scalars

    Writes two files under ``output_dir`` so a later, separate process can
    call ``volume.build_volume()`` from the label-map PNGs and nuclei CSVs
    already written by ``_write_section_outputs``, without re-running
    registration:

        output_dir/section_transforms.pkl -- ``dict[str(path), Affine]``,
            keyed by ``str(path)`` to match ``volume.build_volume``'s lookup
            convention over ``slide_io.load_manifest()`` output.
        output_dir/volume_build_metadata.json -- plain JSON scalars,
            including a ``build_volume_kwargs`` dict splattable straight
            into ``volume.build_volume(**meta["build_volume_kwargs"])``.

    This function never builds a volume itself - only stages the inputs that volume.build_volume() needs.

    Args:
        output_dir: Root output directory (same one passed to run_pipeline).
        slide_paths: Ordered slide paths from ``slide_io.load_manifest``; the
            list index is the section_index used to key ``transforms``.
        transforms: dict from ``registration.build_section_transforms``,
            keyed by section_index. Sections missing from it (e.g.
            registration failed) are skipped, not raised on.
        manifest_path: Path to the manifest CSV, recorded for provenance.
        seg_mpp: Registered/label-map resolution (µm/px); the
            ``section_mpp`` argument ``volume.build_volume()`` expects.
        reg_mpp: Elastic registration resolution (µm/px) the transforms were
            fit at.
        level0_mpp: Native slide resolution (µm/px), or None if uncalibrated.
        section_thickness_um, sampling_interval: cut thickness (microns) and
            imaging interval, or None. ``volume.build_volume_from_run`` reuses
            them when it is not given its own.
        do_non_rigid: Whether elastic refinement was run, for provenance.
        tissue_type: Key into cfg.TISSUE_CLASSES, for provenance.
        segment_first: Whether sections are segmented before registration
            (see ``run_pipeline``). The sections stage reads it back.

    Note:
        ``section_transforms.pkl`` is only safe to load from a trusted run
        directory -- unpickling executes arbitrary code.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_path = {
        str(path): transforms[idx]
        for idx, path in enumerate(slide_paths)
        if idx in transforms
    }
    with open(output_dir / "section_transforms.pkl", "wb") as f:
        pickle.dump(by_path, f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "manifest_csv": str(manifest_path),
        "seg_mpp": seg_mpp,
        "registration_mpp": reg_mpp,
        "level0_mpp": level0_mpp,
        "section_thickness_um": section_thickness_um,
        "sampling_interval": sampling_interval,
        "do_non_rigid": do_non_rigid,
        "tissue_type": tissue_type,
        "segment_first": segment_first,
        "n_sections": len(slide_paths),
        "transforms_pickle": "section_transforms.pkl",
        "label_maps_dir": "label_maps",
        "nuclei_dir": "nuclei",
        "build_volume_kwargs": {
            "section_mpp": seg_mpp,
            "level0_mpp": level0_mpp,
            "registration_mpp": reg_mpp,
        },
        "sections": [
            {
                "section_index": i,
                "path": str(p),
                "has_transform": i in transforms,
            }
            for i, p in enumerate(slide_paths)
        ],
    }
    with open(output_dir / "volume_build_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def _resolve_run_voxel_um(
    voxel_um: float | None,
    section_thickness_um: float | None,
    sampling_interval: int | None,
) -> float:
    """The run's volume resolution: the section spacing when thickness and
    interval are given, else ``voxel_um``, else ``cfg.VOXEL_UM``."""
    if voxel_um is None and section_thickness_um is None and sampling_interval is None:
        voxel_um = cfg.VOXEL_UM
    voxel_um, _ = vol._resolve_voxel_um(voxel_um, section_thickness_um, sampling_interval)
    return voxel_um


def _check_no_volume(output_dir: Path, voxel_um: float) -> Path:
    """Path the volume will be written to; raises if it is already there."""
    volume_path = output_dir / f"volume_{voxel_um:g}um.zarr"
    if volume_path.exists():
        raise FileExistsError(
            f"{volume_path} already exists. Move it or pick another output "
            f"directory; the volume is written at the end of the run."
        )
    return volume_path


def _load_seg_model(seg_model, num_classes, tissue_type: str):
    """Returns ``(model or None, num_classes)``; loads a checkpoint path."""
    if isinstance(seg_model, (str, Path)):
        n = num_classes if num_classes is not None else len(cfg.TISSUE_CLASSES[tissue_type])
        return sg.load_dpt_model(seg_model, n), n
    if seg_model is not None and num_classes is None:
        num_classes = len(cfg.TISSUE_CLASSES[tissue_type])
    return seg_model, num_classes


def _kill_jvm() -> None:
    # The sole call to kill the JVM, at the very end of a process, so that the
    # JVM is only initialized and killed once. Avoids java-level crash / seg fault
    try:
        from valis import registration as _valis_reg
        _valis_reg.kill_jvm()
    except Exception:
        pass


def _register_stack(
    manifest_path: str | Path,
    output_dir: Path,
    slide_paths: list[Path],
    *,
    do_non_rigid: bool,
    seg_mpp: float,
    reg_mpp: float,
    voxel_um: float,
    section_thickness_um: float | None,
    sampling_interval: int | None,
    tissue_type: str,
    strict: bool,
    segment_first: bool = False,
) -> tuple[dict[int, object], list[dict]]:
    """Per-slide tissue masks, then VALIS over the whole stack, then the
    volume-build inputs (transforms + metadata) on disk.

    With ``strict``, failing to write those inputs raises: a staged run's
    later stages read them. Without it, as in ``run_pipeline``, it only warns.

    Returns:
        (section_index -> VALIS slide, per-section results from run_section)
    """
    manifest_rows = [
        {"section_index": i, "path": str(p)}
        for i, p in enumerate(slide_paths)
    ]

    all_results = []
    for section_index, path in enumerate(slide_paths):
        with io.open_slide(path) as slide:
            results = run_section(slide, section_index) # just tissue mask and qc
        all_results.append(results)

    # Stack-level: register all sections together
    slides = reg.register(
        manifest_rows,
        output_dir,
        elastic_mpp=reg_mpp,
        do_non_rigid=do_non_rigid,
    )

    # Persist transforms + volume-build scalars now (before the long
    # warp/segment/detect loop) so if a crash partway through happens, volume can still be built
    try:
        level0_mpp = all_results[0].get("level0_mpp") if all_results else None
        # The transforms' baked-in z is irrelevant: build_volume() overrides
        # it with each section's physical z.
        section_transforms = reg.build_section_transforms(slides, reg_mpp, voxel_um)
        _write_volume_build_inputs(
            output_dir,
            slide_paths,
            section_transforms,
            manifest_path=manifest_path,
            seg_mpp=seg_mpp,
            reg_mpp=reg_mpp,
            level0_mpp=level0_mpp,
            section_thickness_um=section_thickness_um,
            sampling_interval=sampling_interval,
            do_non_rigid=do_non_rigid,
            tissue_type=tissue_type,
            segment_first=segment_first,
        )
        print(
            f"[INFO] Wrote {output_dir / 'section_transforms.pkl'} and "
            f"{output_dir / 'volume_build_metadata.json'}"
        )
    except Exception as exc:
        if strict:
            raise
        print(f"[WARN] Failed to persist volume-build inputs: {exc}")

    return slides, all_results


def _process_section(
    result: dict,
    path: Path,
    valis_slide,
    *,
    output_dir: Path,
    seg_model_obj,
    num_classes: int | None,
    seg_device: str,
    detect_nuclei: bool,
    persist_nuclei_masks: bool,
    seg_mpp: float,
    reg_mpp: float,
    do_non_rigid: bool,
    tissue_type: str,
    segment_first: bool = False,
) -> None:
    """Warp, segment and detect nuclei in one section, write its outputs and
    release them from ``result`` (mutated in place).

    ``result`` needs 'section_index' and 'level0_shape_rc'. ``valis_slide`` is
    the section's VALIS slide object, or None if registration dropped it.

    With ``segment_first`` the ORIGINAL slide is segmented at the training
    resolution, the raw label map is kept under ``label_maps_raw/``, and it is
    warped (nearest-neighbour) onto the registered canvas, so
    ``label_maps/`` holds the same frame either way. Otherwise the registered
    OME-TIFF is segmented, as before.
    """
    idx = result["section_index"]
    registered_dir = output_dir / "registered"
    registered_dir.mkdir(parents=True, exist_ok=True)
    nuclei_dir = output_dir / "nuclei"
    if persist_nuclei_masks:
        nuclei_dir.mkdir(parents=True, exist_ok=True)

    if segment_first and seg_model_obj is not None:
        assert num_classes is not None
        raw_labels, raw_mpp = _segment_raw_section(
            path,
            seg_model_obj,
            num_classes=num_classes,
            section_idx=idx,
            seg_device=seg_device,
        )
        _save_label_png(raw_labels, output_dir / "label_maps_raw" / f"{idx:04d}_labels.png")
        result["raw_label_mpp"] = raw_mpp
        if valis_slide is not None:
            result["label_map"] = reg.warp_label_map_to_registered(
                valis_slide,
                raw_labels,
                reg_mpp=reg_mpp,
                target_mpp=seg_mpp,
                non_rigid=do_non_rigid,
            )
        del raw_labels

    # Warp the section straight from its original slide to seg_mpp: the
    # registered H&E, segmented here unless segment_first already did it
    if valis_slide is None:
        result["registered_path"] = None
    else:
        out_path = reg.warp_and_save_section(
            valis_slide,
            path,
            registered_dir / f"{idx:04d}.ome.tiff",
            target_mpp=seg_mpp,
            reg_mpp=reg_mpp,
            non_rigid=do_non_rigid,
        )
        result["registered_path"] = out_path

        # Segmentation runs before nuclear detection so it always gets clean GPU state

        if seg_model_obj is not None and not segment_first:
            # num_classes is always set to an int alongside seg_model_obj
            # in _load_seg_model -- assert narrows it for the type checker
            assert num_classes is not None
            # Tissue mask computed directly from the already-registered
            # OME-TIFF at _MASK_MPP. No manual resize belongs here
            # either way: predict_section upsamples this to the seg_mpp
            # label grid with order=0 nearest-neighbour (categorical
            # data; bilinear would invent invalid class values).
            #
            # Deliberate trade-off vs the previous recipe (kept here for
            # history, not as a TODO): this no longer warps a mask from
            # the *original* slide through VALIS, so it no longer needs
            # the live valis_slide object for masking, and no longer
            # excludes VALIS-invented canvas padding outside the rotated
            # section (previously done via an all-255 valid_mask warp --
            # see git history). A non-black VALIS border fill can
            # therefore be misclassified as tissue by make_tissue_mask's
            # brightness heuristic. Accepted intentionally for the
            # simplicity of not needing a live valis_slide object here.
            with io.open_slide(out_path) as slide:
                viz_rgb = pr.downsample_to_mpp(slide, _MASK_MPP)
            tissue_mask = pr.make_tissue_mask(viz_rgb)
            result["label_map"] = sg.predict_section(
                out_path,
                seg_model_obj,
                tissue_mask,
                num_classes=num_classes,
                section_idx=idx,
                seg_mpp=seg_mpp,
                device=seg_device,
            )

    # Nuclear detection runs after segmentation. runs on the original slide, independent of registration success
    # as its use is for downstream analysis

    if detect_nuclei:
        output_zarr = nuclei_dir / f"{idx:04d}.zarr" if persist_nuclei_masks else None
        result["nuclei"] = nd.detect_nuclei_section(
            path,
            seg_mpp=seg_mpp,
            gpu=(seg_device != "cpu"),
            output_zarr=output_zarr,
        )
        # Centroids are in the unregistered slide's level-0 pixels. Map
        # them into the registered canvas while the VALIS slide object is
        # live, through the same rigid + elastic warp the image (and so
        # the label map) went through. build_volume uses these columns
        # instead of its rigid-only Affine.
        if valis_slide is not None:
            _add_registered_coordinates(
                result["nuclei"],
                valis_slide,
                level0_shape_rc=result["level0_shape_rc"],
                reg_mpp=reg_mpp,
                target_mpp=seg_mpp,
                non_rigid=do_non_rigid,
            )

    # Write whatever this section produced to disk now so that crash (OOM/walltime) on a later section doesn't lose it
    # Then release the heavy RAM values (label map ndarray, nuclei DataFrame) back to on-disk references, so no more
    # than one section output is ever held at once.
    written = _write_section_outputs(result, output_dir, tissue_type=tissue_type)
    _release_section_payload(result, written)


def _build_and_quantify(
    output_dir: Path,
    manifest_path: str | Path,
    *,
    voxel_um: float,
    volume_path: Path,
    section_thickness_um: float | None,
    sampling_interval: int | None,
    detect_nuclei: bool,
    tissue_type: str,
) -> None:
    """3D volume, then quantification. Built from the per-section files on
    disk, so it is exactly what volume.build_volume_from_run reproduces from
    a finished directory."""
    parquet_path = output_dir / f"nuclei_{voxel_um:g}um.parquet" if detect_nuclei else None
    sdata = vol.build_volume_from_run(
        output_dir,
        manifest_csv=manifest_path,
        target_voxel_um=voxel_um,
        section_thickness_um=section_thickness_um,
        sampling_interval=sampling_interval,
        include_nuclei=detect_nuclei,
        output_path=volume_path,
        nuclear_parquet_path=parquet_path,
    )
    print(f"[INFO] Wrote {volume_path}" + (f" and {parquet_path}" if parquet_path else ""))

    qt.run_quantification(
        sdata,
        output_dir / "quantification",
        organ=tissue_type,
        voxel_um=voxel_um,
        section_thickness_um=section_thickness_um,
        sampling_interval=sampling_interval,
    )


def run_pipeline(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    do_non_rigid: bool = True,
    seg_model=None,
    num_classes=None,
    tissue_type: str = "HGSC", # update for generalized run, fine for now
    seg_device: str = "cuda", # update for generalized run, fine for now
    detect_nuclei: bool = True,
    persist_nuclei_masks: bool = False,
    seg_mpp: float = cfg.SEG_MPP,
    reg_mpp: float = cfg.REG_MPP,
    voxel_um: float | None = None,
    section_thickness_um: float | None = None,
    sampling_interval: int | None = None,
    segment_first: bool = False,
) -> list[dict]:
    """Run the full pipeline over all sections in the manifest, through the 3D
    volume and its quantification report.

    Each slide is opened, processed, and closed before the next one is opened,
    keeping file-descriptor usage at 1 regardless of stack size. Registration
    runs as a stack-level operation after all sections are processed
    individually (VALIS needs all slides together). Once every section is on
    disk, the volume is built from those files and quantified. Nothing is
    rendered or opened for viewing: ``visualization.view_volume`` on the
    written zarr does that, locally.

    Args:
        manifest_path: Path to a manifest CSV from slide_io.create_manifest.
        output_dir:    Directory for VALIS outputs (thumbnails, transforms, logs).
        do_non_rigid:  Run elastic refinement after rigid pass.
        seg_model:     A loaded segmentation nn.Module, OR a str/Path checkpoint
            to load via segmentation.load_dpt_model, OR None to skip
            segmentation inference entirely.
        num_classes:   Number of tissue classes for segmentation. If None and
            seg_model is given, derived from cfg.TISSUE_CLASSES[tissue_type].
        tissue_type:   Key into cfg.TISSUE_CLASSES used to derive num_classes
            when num_classes is None.
        seg_device:    Device string passed to segmentation.predict_section.
        detect_nuclei: Run nuclear detection on each section's original slide.
        persist_nuclei_masks: When True, persist per-section instance masks to
            zarr under output_dir/nuclei/.
        seg_mpp:       Registered output resolution (µm/px), shared by the
            streaming warp, ``predict_section``, and ``detect_nuclei_section``
            so they can't silently diverge.
        reg_mpp:       Elastic registration resolution (µm/px). Must match
            what ``warp_and_save_section`` uses to derive its output canvas
            scale (``reg_mpp / seg_mpp``).
        voxel_um:      Isotropic resolution (µm/voxel) of the 3D volume and
            its quantification, used only without ``section_thickness_um``/
            ``sampling_interval``. None means ``cfg.VOXEL_UM``.
        section_thickness_um: Thickness each section was cut at (µm). Give
            with ``sampling_interval``.
        sampling_interval: Every how many cut sections one was stained and
            imaged (1 = every section, 2 = every second). With
            ``section_thickness_um`` it sets the section spacing, which is
            the voxel size (4 µm x 2 = 8 µm voxels), and the stereological
            count correction. Without them, the correction treats sections
            as ``voxel_um`` thick with every one imaged.
        segment_first: Segment each ORIGINAL slide at the training
            resolution (the pyramid level nearest ``cfg.SEG_TRAIN_MPP``),
            then warp its label map nearest-neighbour onto the registered
            canvas, instead of segmenting the registered OME-TIFF. The raw
            label maps are kept in output_dir/label_maps_raw/. Registration
            itself, the registered H&E, nuclei, the volume and its
            quantification are unchanged.

    Returns:
        List of per-section result dicts in section order. Each dict
        contains 'section_index', 'tissue_mask_80', 'img_80', 'level0_mpp',
        'level0_shape_rc' and 'registered_path', plus:
            'label_map' -- the on-disk label-PNG Path, or None if no
                segmentation ran for this section. Never an ndarray.
            'nuclei'    -- {'n_nuclei': int, 'path': Path}, or None if no
                nuclear detection ran. Never a DataFrame.
        Both are released to their on-disk form as soon as each section is
        written, so RAM doesn't grow with section count over a long run.
        Nuclei tables of registered sections carry 'reg_y_um'/'reg_x_um',
        the centroids in the registered canvas (see
        ``registration.warp_points_to_registered``).
        Per-section outputs land in output_dir/label_maps/ and
        output_dir/nuclei/ as each section completes, so a crash mid-run
        leaves completed sections on disk. Immediately after registration,
        output_dir/section_transforms.pkl and
        output_dir/volume_build_metadata.json are also written, so the
        volume can be rebuilt offline with ``volume.build_volume_from_run``.

        With segmentation, the run ends by writing
            output_dir/volume_{voxel_um}um.zarr -- SpatialData volume (labels,
                plus the nuclei table when nuclei were detected).
            output_dir/nuclei_{voxel_um}um.parquet -- merged nuclear table.
            output_dir/quantification/ -- result tables and report.md.
        Without segmentation there are no label maps, so no volume either.

    Raises:
        FileExistsError: the volume zarr already exists in ``output_dir``.
        ValueError: only one of ``section_thickness_um``/``sampling_interval``
            is given, or ``voxel_um`` disagrees with their spacing.
        Both are checked before any work, not after hours of it.
    """
    voxel_um = _resolve_run_voxel_um(voxel_um, section_thickness_um, sampling_interval)

    output_dir = Path(output_dir)
    volume_path = output_dir / f"volume_{voxel_um:g}um.zarr"
    if seg_model is not None:
        _check_no_volume(output_dir, voxel_um)

    slide_paths = io.load_manifest(manifest_path)
    seg_model_obj, num_classes = _load_seg_model(seg_model, num_classes, tissue_type)

    slides, all_results = _register_stack(
        manifest_path,
        output_dir,
        slide_paths,
        do_non_rigid=do_non_rigid,
        seg_mpp=seg_mpp,
        reg_mpp=reg_mpp,
        voxel_um=voxel_um,
        section_thickness_um=section_thickness_um,
        sampling_interval=sampling_interval,
        tissue_type=tissue_type,
        strict=False,
        segment_first=segment_first,
    )

    for result, path in zip(all_results, slide_paths):
        _process_section(
            result,
            path,
            slides.get(result["section_index"]),
            output_dir=output_dir,
            seg_model_obj=seg_model_obj,
            num_classes=num_classes,
            seg_device=seg_device,
            detect_nuclei=detect_nuclei,
            persist_nuclei_masks=persist_nuclei_masks,
            seg_mpp=seg_mpp,
            reg_mpp=reg_mpp,
            do_non_rigid=do_non_rigid,
            tissue_type=tissue_type,
            segment_first=segment_first,
        )

    _kill_jvm()

    if seg_model_obj is None:
        print("[INFO] Segmentation skipped: no label maps, so no 3D volume or quantification.")
        return all_results

    _build_and_quantify(
        output_dir,
        manifest_path,
        voxel_um=voxel_um,
        volume_path=volume_path,
        section_thickness_um=section_thickness_um,
        sampling_interval=sampling_interval,
        detect_nuclei=detect_nuclei,
        tissue_type=tissue_type,
    )

    return all_results


# Staged run. run_pipeline above does the same work in one process; these split
# it at the points where nothing is held in memory between the parts, so each
# can run as its own Slurm job:
#
#   1. run_register_stage  (CPU)        VALIS over the whole stack
#   2. run_sections_stage  (GPU array)  warp + segment + nuclei, a share of
#                                       the sections per task
#   3. run_volume_stage    (CPU)        3D volume + quantification
#
# Stage 1 writes the registration (VALIS's pickled registrar) and
# volume_build_metadata.json; stages 2 and 3 read their settings back from
# there, so the stages cannot disagree on the manifest, resolutions or
# section spacing.


def _read_run_metadata(output_dir: Path) -> dict:
    path = output_dir / "volume_build_metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the register stage into {output_dir} first."
        )
    with open(path) as f:
        return json.load(f)


def run_register_stage(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    do_non_rigid: bool = True,
    tissue_type: str = "HGSC",
    seg_mpp: float = cfg.SEG_MPP,
    reg_mpp: float = cfg.REG_MPP,
    voxel_um: float | None = None,
    section_thickness_um: float | None = None,
    sampling_interval: int | None = None,
    segment_first: bool = False,
) -> list[dict]:
    """Stage 1 of a staged run: register the stack and record the run's
    settings, ``segment_first`` included. Arguments as for ``run_pipeline``.

    Raises:
        FileExistsError: the volume zarr already exists in ``output_dir``.
        ValueError: as ``run_pipeline``, before any work.
    """
    voxel_um = _resolve_run_voxel_um(voxel_um, section_thickness_um, sampling_interval)
    output_dir = Path(output_dir)
    _check_no_volume(output_dir, voxel_um)

    slide_paths = io.load_manifest(manifest_path)
    _, all_results = _register_stack(
        manifest_path,
        output_dir,
        slide_paths,
        do_non_rigid=do_non_rigid,
        seg_mpp=seg_mpp,
        reg_mpp=reg_mpp,
        voxel_um=voxel_um,
        section_thickness_um=section_thickness_um,
        sampling_interval=sampling_interval,
        tissue_type=tissue_type,
        strict=True,
        segment_first=segment_first,
    )
    _kill_jvm()
    return all_results


def run_sections_stage(
    output_dir: str | Path,
    *,
    task_index: int = 0,
    n_tasks: int = 1,
    seg_model=None,
    num_classes=None,
    seg_device: str = "cuda",
    detect_nuclei: bool = True,
    persist_nuclei_masks: bool = False,
) -> list[dict]:
    """Stage 2 of a staged run: warp, segment and detect nuclei in this task's
    share of the sections (``task_sections``).

    Reloads the registration stage 1 saved. Sections whose outputs are already
    on disk are skipped, so a failed or timed-out task can simply be
    resubmitted. Model arguments as for ``run_pipeline``; the manifest,
    tissue type, resolutions, do_non_rigid and segment_first come from
    stage 1 (runs registered before segment_first existed segment the
    registered image, as they did then).

    Returns:
        Result dicts, as ``run_pipeline``'s, for the sections processed now.
    """
    output_dir = Path(output_dir)
    meta = _read_run_metadata(output_dir)
    tissue_type = meta["tissue_type"]
    slide_paths = io.load_manifest(meta["manifest_csv"])

    mine = task_sections(len(slide_paths), task_index, n_tasks)
    todo = [
        idx for idx in mine
        if not _section_done(output_dir, idx, segment=seg_model is not None,
                             detect_nuclei=detect_nuclei)
    ]
    print(f"[INFO] Task {task_index} of {n_tasks}: sections {list(mine)}, "
          f"{len(mine) - len(todo)} already done")
    if not todo:
        return []

    seg_model_obj, num_classes = _load_seg_model(seg_model, num_classes, tissue_type)
    slides = reg.load_registered_slides(output_dir, todo)

    results = []
    for idx in todo:
        path = slide_paths[idx]
        with io.open_slide(path) as slide:
            level0_w, level0_h = slide.level_dimensions[0]
        result = {"section_index": idx, "level0_shape_rc": (int(level0_h), int(level0_w))}
        _process_section(
            result,
            path,
            slides.get(idx),
            output_dir=output_dir,
            seg_model_obj=seg_model_obj,
            num_classes=num_classes,
            seg_device=seg_device,
            detect_nuclei=detect_nuclei,
            persist_nuclei_masks=persist_nuclei_masks,
            seg_mpp=meta["seg_mpp"],
            reg_mpp=meta["registration_mpp"],
            do_non_rigid=meta["do_non_rigid"],
            tissue_type=tissue_type,
            segment_first=meta.get("segment_first", False),
        )
        results.append(result)

    _kill_jvm()
    return results


def run_volume_stage(
    output_dir: str | Path,
    *,
    voxel_um: float | None = None,
    detect_nuclei: bool = True,
) -> None:
    """Stage 3 of a staged run: build the 3D volume and quantify it, once
    every section has its label map.

    ``voxel_um`` only matters when stage 1 had no section thickness and
    interval; pass the same value stage 1 got.

    Raises:
        FileNotFoundError: a section has no label map, listing which.
        FileExistsError: the volume zarr already exists.
    """
    output_dir = Path(output_dir)
    meta = _read_run_metadata(output_dir)
    thickness_um = meta.get("section_thickness_um")
    interval = meta.get("sampling_interval")
    voxel_um = _resolve_run_voxel_um(voxel_um, thickness_um, interval)
    volume_path = _check_no_volume(output_dir, voxel_um)

    n = len(io.load_manifest(meta["manifest_csv"]))
    missing = [
        idx for idx in range(n)
        if not (output_dir / "label_maps" / f"{idx:04d}_labels.png").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"No label map for sections {missing}. Rerun the sections stage "
            f"for the tasks that own them; finished sections are skipped."
        )

    _build_and_quantify(
        output_dir,
        meta["manifest_csv"],
        voxel_um=voxel_um,
        volume_path=volume_path,
        section_thickness_um=thickness_um,
        sampling_interval=interval,
        detect_nuclei=detect_nuclei,
        tissue_type=meta["tissue_type"],
    )
