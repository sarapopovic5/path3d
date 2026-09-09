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


# DATA FLOW: one slide is open at a time so as to not overload RAM


def run_section(slide: io.SlideReader, section_index: int) -> dict:
    """Process one section through its per-slide stages (tissue mask + QC prep).

    QC pass/fail itself is decided later, by pr.flag_qc_outliers on the
    full stack of results.

    Args:
        slide: Open SlideReader for this section (caller owns it).
        section_index: 0-based z-index of this section (from manifest).

    Returns:
        Per-section results dict, including ``level0_mpp`` (native µm/px) so later stages dont need to reopen the slide
    """
    results: dict = {"section_index": section_index}

    # tissue mask + QC (80 mpp — coarsest, cheapest)
    img_80 = pr.downsample_to_mpp(slide, 80.0)
    results["tissue_mask_80"] = pr.make_tissue_mask(img_80)
    results["img_80"] = img_80
    results["level0_mpp"] = slide.get_mpp()

    return results


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
        label_map_path = label_map_dir / f"{idx:04d}_labels.png"
        Image.fromarray(label_map_u8).save(label_map_path)
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
        df.to_csv(nuclei_csv_path, index=False)
        written["nuclei_csv_path"] = nuclei_csv_path

    return written


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
    section_thickness_um: float,
    do_non_rigid: bool,
    tissue_type: str,
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
        section_thickness_um: Z spacing (microns), recorded for provenance
            only -- ``build_volume`` derives its own z from the manifest.
        do_non_rigid: Whether elastic refinement was run, for provenance.
        tissue_type: Key into cfg.TISSUE_CLASSES, for provenance.

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
        "do_non_rigid": do_non_rigid,
        "tissue_type": tissue_type,
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
) -> list[dict]:
    """Run the full pipeline over all sections in the manifest.

    Each slide is opened, processed, and closed before the next one is opened,
    keeping file-descriptor usage at 1 regardless of stack size. Registration
    runs as a stack-level operation after all sections are processed
    individually (VALIS needs all slides together).

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

    Returns:
        List of per-section result dicts in section order. Each dict
        contains 'section_index', 'tissue_mask_80', 'img_80', 'level0_mpp',
        and 'registered_path', plus:
            'label_map' -- the on-disk label-PNG Path, or None if no
                segmentation ran for this section. Never an ndarray.
            'nuclei'    -- {'n_nuclei': int, 'path': Path}, or None if no
                nuclear detection ran. Never a DataFrame.
        Both are released to their on-disk form as soon as each section is
        written, so RAM doesn't grow with section count over a long run.
        Per-section outputs land in output_dir/label_maps/ and
        output_dir/nuclei/ as each section completes, so a crash mid-run
        leaves completed sections on disk. Immediately after registration,
        output_dir/section_transforms.pkl and
        output_dir/volume_build_metadata.json are also written, so
        ``volume.build_volume()`` can be called OFFLINE afterwards.
        ``run_pipeline`` never calls ``volume.build_volume()`` itself
    """
    slide_paths = io.load_manifest(manifest_path)
    all_results = []

    
    seg_model_obj = seg_model
    if isinstance(seg_model, (str, Path)):
        n = num_classes if num_classes is not None else len(cfg.TISSUE_CLASSES[tissue_type])
        seg_model_obj = sg.load_dpt_model(seg_model, n)
        num_classes = n
    elif seg_model is not None and num_classes is None:
        num_classes = len(cfg.TISSUE_CLASSES[tissue_type])

    manifest_rows = [
        {"section_index": i, "path": str(p)}
        for i, p in enumerate(slide_paths)
    ]

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
    # warp/segment/detect loop below) so if a crash partway through happens, volume can still be built 
    # Convenience artifact only -failures are logged (not raised) so they can't abort the run
    try:
        level0_mpp = all_results[0].get("level0_mpp") if all_results else None
        # section_thickness_um left at build_section_transforms's 12.0 default
        # (not the user's actual thickness) -- harmless, since build_volume()
        # recomputes z from the manifest and ignores this baked-in value.
        section_transforms = reg.build_section_transforms(slides, reg_mpp) 
        _write_volume_build_inputs(
            Path(output_dir),
            slide_paths,
            section_transforms,
            manifest_path=manifest_path,
            seg_mpp=seg_mpp,
            reg_mpp=reg_mpp,
            level0_mpp=level0_mpp,
            section_thickness_um=12.0,
            do_non_rigid=do_non_rigid,
            tissue_type=tissue_type,
        )
        print(
            f"[INFO] Wrote {Path(output_dir) / 'section_transforms.pkl'} and "
            f"{Path(output_dir) / 'volume_build_metadata.json'}"
        )
    except Exception as exc:
        print(f"[WARN] Failed to persist volume-build inputs: {exc}")

    # Warp each section straight from its original slide to seg_mpp and save for segmentation
    
    registered_dir = Path(output_dir) / "registered"
    registered_dir.mkdir(parents=True, exist_ok=True)

    nuclei_dir = Path(output_dir) / "nuclei"
    if persist_nuclei_masks:
        nuclei_dir.mkdir(parents=True, exist_ok=True)

    for result, path in zip(all_results, slide_paths):
        idx = result["section_index"]
        valis_slide = slides.get(idx)

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
           
            if seg_model_obj is not None:
                # num_classes is always set to an int alongside seg_model_obj
                # above -- assert narrows it for the type checker
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

        # Write whatever this section produced to disk now so that crash (OOM/walltime) on a later section doesn't lose it
        # Then release the heavy RAM values (label map ndarray, nuclei DataFrame) back to on-disk references, so no more
        # than one section output is ever in all_results at once.
        written = _write_section_outputs(result, Path(output_dir), tissue_type=tissue_type)
        _release_section_payload(result, written)


    # the sole call to kill the JVM, at the very end of the pipeline, so that JVM is only initialized and killed once
    # Avoids java-level crash / seg fault
    try:
        from valis import registration as _valis_reg
        _valis_reg.kill_jvm()
    except Exception:
        pass

    return all_results
