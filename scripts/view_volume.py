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


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/view_volume.py VOLUME.zarr [TISSUE_TYPE]")
    path = sys.argv[1]
    tissue_type = sys.argv[2] if len(sys.argv) > 2 else tissue_type_of(path)
    print(f"tissue type: {tissue_type}", flush=True)

    with TRACE.open("w") as fh:
        fh.write("t_s,rss_gib,avail_gib,stage\n")
        threading.Thread(target=_sampler, args=(fh,), daemon=True).start()
        try:
            stage("01 view_volume (read_zarr + layers)")
            view_volume(path, tissue_type=tissue_type)

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
