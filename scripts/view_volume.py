"""Ad-hoc napari launcher that tracks this process's RSS while viewing.

``path3d.visualization.view_volume`` deliberately does no memory tracking.
This keeps that diagnostic out of the package: useful because the full
volumes are large enough that a viewer session can exhaust memory, but not
something an installed viewer should be doing.

A daemon thread samples RSS every INTERVAL seconds. New peaks print live;
a per-stage table and an ASCII timeline print once the window is closed.
Samples are appended to ``rss_trace.csv`` in the current directory and
flushed per sample, so a crash or force-quit still leaves a usable trace.

    python scripts/view_volume.py path/to/volume_dpt_8um.zarr
    python scripts/view_volume.py path/to/volume_niches_32um.zarr

The palette and class names are picked from the volume itself: a niche volume
(from ``python -m path3d.niches.run``) records ``niche_classes`` in its root
attributes and opens as ``HGSC_niches``; anything else opens as ``HGSC``.
Pass a tissue type as a second argument to override.

A niche volume coarser than 8 um (e.g. 32 um cubes) is shown smoothed: its
probabilities are interpolated to 8 um cubes and argmaxed, in memory, for
display only (nothing is written). Add ``--raw`` to see the stored cubes. An
8 um volume is shown as stored.

Requires psutil (``pip install path3d[viz]`` does not pull it in; ``pip
install psutil``).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import napari
import psutil

from path3d.visualization import view_volume

TRACE = Path("rss_trace.csv")
INTERVAL = 0.1
PEAK_REPORT_STEP = 0.25  # GiB -- print a live line each time the peak grows this much
# Niche volumes coarser than this are drawn at it (32 um -> 8 um), interpolated
# from their probabilities. Volumes already this fine are shown as stored, as
# are all volumes with --raw.
DISPLAY_UM = 8.0

_PROC = psutil.Process()
_T0 = time.monotonic()
_stage = "00 start"
_stop = threading.Event()
_samples: list[tuple[float, float, float, str]] = []


def _sampler(fh) -> None:
    """Poll RSS until told to stop. Runs in a daemon thread."""
    reported = 0.0
    while not _stop.is_set():
        t = time.monotonic() - _T0
        rss = _PROC.memory_info().rss / 2**30
        avail = psutil.virtual_memory().available / 2**30
        _samples.append((t, rss, avail, _stage))
        fh.write(f"{t:.3f},{rss:.4f},{avail:.4f},{_stage}\n")
        fh.flush()
        if rss >= reported + PEAK_REPORT_STEP:
            reported = rss
            print(f"[{t:7.1f}s] peak rss {rss:6.2f} GiB "
                  f"(system available {avail:5.2f} GiB)  <- {_stage}", flush=True)
        _stop.wait(INTERVAL)


def stage(name: str) -> None:
    """Mark the start of a stage and report RSS at the transition."""
    global _stage
    _stage = name
    rss = _PROC.memory_info().rss / 2**30
    print(f"[{time.monotonic() - _T0:7.1f}s] {name:<40} rss={rss:6.2f} GiB", flush=True)


def summarise() -> None:
    """Per-stage enter/peak/exit table, overall peak, and an ASCII timeline."""
    if not _samples:
        return
    peak = max(s[1] for s in _samples)
    floor = min(s[2] for s in _samples)

    print("\n--- per-stage RSS (GiB) ---")
    print(f"{'stage':<42}{'enter':>8}{'peak':>8}{'exit':>8}{'delta':>8}")
    order: list[str] = []
    by_stage: dict[str, list[float]] = {}
    for _, rss, _, name in _samples:
        if name not in by_stage:
            by_stage[name] = []
            order.append(name)
        by_stage[name].append(rss)
    for name in order:
        v = by_stage[name]
        print(f"{name:<42}{v[0]:8.2f}{max(v):8.2f}{v[-1]:8.2f}{v[-1] - v[0]:+8.2f}")

    total = psutil.virtual_memory().total / 2**30
    swap = psutil.swap_memory()
    print(f"\nPEAK RSS             = {peak:.2f} GiB")
    print(f"min system available = {floor:.2f} GiB of {total:.1f} GiB total")
    print(f"swap used now        = {swap.used / 2**30:.2f} GiB")

    # Timeline: columns are equal slices of wall clock, cell = max RSS in slice.
    print("\n--- RSS over time ---")
    cols, rows = 72, 12
    span = _samples[-1][0] or 1.0
    buckets = [0.0] * cols
    for t, rss, _, _ in _samples:
        i = min(cols - 1, int(t / span * cols))
        buckets[i] = max(buckets[i], rss)
    for r in range(rows, 0, -1):
        level = peak * r / rows
        print(f"{level:6.1f} |" + "".join("#" if b >= level else " " for b in buckets))
    print(f"{0.0:6.1f} +" + "-" * cols)
    print(f"       0s{' ' * (cols - 12)}{span:.0f}s")


def tissue_type_of(path: str | Path) -> str:
    """``HGSC_niches`` for a niche volume, else ``HGSC``, from the zarr root attrs."""
    try:
        attrs = json.loads((Path(path) / ".zattrs").read_text())
    except (OSError, ValueError):
        return "HGSC"
    return "HGSC_niches" if "niche_classes" in attrs else "HGSC"


def _smooth_niche_labels(sdata, factor: int):
    """Redraw a niche volume's labels on a ``factor``x finer grid, smoothly.

    Interpolates ``images["niche_probabilities"]`` trilinearly -- NaN-aware,
    so off-tissue never bleeds in -- then takes the argmax per fine voxel.
    For display only: it is the same prediction with smooth boundaries
    instead of 32 um blocks, and it is never written anywhere.

    Coordinates are centre-aligned: fine index ``i`` samples coarse position
    ``(i + 0.5) / factor - 0.5``, which is where ``skimage.transform.resize``
    puts pixel centres, so z and in-plane agree.
    """
    import numpy as np
    from skimage.transform import resize
    from spatialdata import SpatialData
    from spatialdata.models import Labels3DModel
    from spatialdata.transformations import Affine, get_transformation

    from path3d.niches.rasterize import argmax_labels

    element = sdata.images["niche_probabilities"]
    probs = np.asarray(element.data, dtype=np.float32)  # (c, z, y, x)
    classes = [str(c) for c in element.coords["c"].values]
    n_classes, nz, ny, nx = probs.shape
    valid = np.isfinite(probs).all(axis=0)
    filled = np.where(valid, probs, 0.0).astype(np.float32)
    voxel_um = float(
        get_transformation(element, "microns_3d").to_affine_matrix(
            input_axes=("z", "y", "x"), output_axes=("z", "y", "x")
        )[0, 0]
    )

    out_hw = (ny * factor, nx * factor)

    def up(plane):
        return resize(plane, out_hw, order=1, mode="edge",
                      anti_aliasing=False, preserve_range=True).astype(np.float32)

    labels = np.zeros((nz * factor, *out_hw), np.uint8)
    for i in range(nz * factor):
        c = min(max((i + 0.5) / factor - 0.5, 0.0), nz - 1.0)
        lo = int(np.floor(c))
        hi = min(lo + 1, nz - 1)
        t = c - lo
        weight = up((1 - t) * valid[lo] + t * valid[hi])
        plane = np.empty((n_classes, *out_hw), np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            for k in range(n_classes):
                plane[k] = up((1 - t) * filled[k, lo] + t * filled[k, hi]) / weight
        plane[:, weight < 0.5] = np.nan
        labels[i] = argmax_labels(plane, classes)

    fine_um = voxel_um / factor
    affine = Affine(np.diag([fine_um, fine_um, fine_um, 1.0]),
                    input_axes=("z", "y", "x"), output_axes=("z", "y", "x"))
    labels_el = Labels3DModel.parse(
        labels, dims=("z", "y", "x"), transformations={"microns_3d": affine}
    ).chunk({"z": 1, "y": 512, "x": 512})
    print(f"smoothed display: {voxel_um:g} um -> {fine_um:g} um cubes, "
          f"shape {labels.shape}", flush=True)
    return SpatialData(labels={"tissue_labels": labels_el}, tables=dict(sdata.tables))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    raw = "--raw" in sys.argv
    if not args:
        sys.exit("usage: python scripts/view_volume.py VOLUME.zarr [TISSUE_TYPE] [--raw]")
    path = args[0]
    tissue_type = args[1] if len(args) > 1 else tissue_type_of(path)
    print(f"tissue type: {tissue_type}", flush=True)

    with TRACE.open("w") as fh:
        fh.write("t_s,rss_gib,avail_gib,stage\n")
        threading.Thread(target=_sampler, args=(fh,), daemon=True).start()
        try:
            volume = path
            if tissue_type == "HGSC_niches" and not raw:
                import spatialdata

                stage("01a smooth niche labels")
                sdata = spatialdata.read_zarr(path)
                voxel_um = sdata.attrs.get("niche_voxel_um_zyx", [DISPLAY_UM])[0]
                factor = int(round(voxel_um / DISPLAY_UM))
                if factor > 1 and "niche_probabilities" in sdata.images:
                    volume = _smooth_niche_labels(sdata, factor)
            stage("01 view_volume (read_zarr + layers)")
            view_volume(volume, tissue_type=tissue_type)

            # Blocks until the window is closed. Rotating/zooming and toggling
            # the hidden class_ layers happens inside this stage, so the live
            # peak lines show what each interaction costs.
            stage("02 napari.run (interactive)")
            napari.run()

            stage("03 window closed")
            time.sleep(0.5)
        finally:
            _stop.set()
            time.sleep(0.3)

    summarise()
    print(f"\ntrace: {TRACE.resolve()}")


if __name__ == "__main__":
    main()
