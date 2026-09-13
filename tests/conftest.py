"""Shared pytest fixtures.

Real slide data never lives in this repository and its location is never
hardcoded -- path3d is a public repo, and the slides are patient-derived.
Tests that need a real image take its path at runtime and skip when it is not
supplied, so ``pytest`` stays green on a clean checkout with no data.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_OME_TIFF_ENV = "PATH3D_TEST_OME_TIFF"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--ome-tiff",
        action="store",
        default=None,
        metavar="PATH",
        help=(
            "Path to one registered OME-TIFF to run integration tests against. "
            f"May also be set via the {_OME_TIFF_ENV} environment variable."
        ),
    )
    parser.addoption(
        "--run-encoder",
        action="store_true",
        default=False,
        help=(
            "Also run the UNI2-h forward pass in integration tests. Requires "
            "Hugging Face access to the gated MahmoodLab/UNI2-h and downloads "
            "~2.5 GB on first use."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: needs real data and/or the UNI2-h encoder"
    )


@pytest.fixture(scope="session")
def ome_tiff_path(request: pytest.FixtureRequest) -> Path:
    """One registered OME-TIFF, from ``--ome-tiff`` or ``$PATH3D_TEST_OME_TIFF``.

    Skips the test when neither is set, or when the path does not exist.
    """
    raw = request.config.getoption("--ome-tiff") or os.environ.get(_OME_TIFF_ENV)
    if not raw:
        pytest.skip(
            f"no OME-TIFF supplied; pass --ome-tiff PATH or set {_OME_TIFF_ENV}"
        )
    path = Path(raw).expanduser()
    if not path.exists():
        pytest.skip(f"OME-TIFF not found: {path}")
    return path


@pytest.fixture(scope="session")
def run_encoder(request: pytest.FixtureRequest) -> bool:
    """Whether to exercise the gated UNI2-h encoder."""
    return bool(request.config.getoption("--run-encoder"))
