"""Shared pytest fixtures.

Everything here is synthetic and written to ``tmp_path``. path3d is a public
repo and the real slides are patient-derived, so no test may depend on slide
data being present.

Run from the repo root after ``pip install -e ".[dev]"``::

    pytest tests
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def make_manifest(tmp_path: Path):
    """Factory: write an ``n``-row manifest CSV into ``tmp_path``.

    Rows use the ``slide_io.create_manifest`` header (section_index, filename,
    path) and point at ``s{i}.czi`` files that do not exist; code that only
    reads the manifest (``load_manifest``, ``build_volume``) never opens them.
    Extra columns are passed as ``name=[one value per row]``, e.g.
    ``make_manifest(3, thickness_um=[8, 8, 8])``.
    """

    def _make(n: int, **columns: list) -> Path:
        for name, values in columns.items():
            if len(values) != n:
                raise ValueError(f"{name} has {len(values)} values for {n} rows")
        path = tmp_path / "manifest.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["section_index", "filename", "path", *columns])
            for i in range(n):
                extra = [values[i] for values in columns.values()]
                writer.writerow([i, f"s{i}.czi", str(tmp_path / f"s{i}.czi"), *extra])
        return path

    return _make


@pytest.fixture
def make_label_maps():
    """Factory: one ``(H, W)`` uint8 label map per manifest row.

    Keyed by ``str(path)``, as ``volume.build_volume`` expects. Section k is
    filled with class ``k % 5`` (the five HGSC classes), so a plane's value
    says which section it came from.
    """

    def _make(manifest: Path, shape: tuple[int, int] = (40, 40)) -> dict[str, np.ndarray]:
        from path3d.slide_io import load_manifest

        return {
            str(path): np.full(shape, k % 5, dtype=np.uint8)
            for k, path in enumerate(load_manifest(manifest))
        }

    return _make
