#!/bin/bash
# Build the path3d venv on a Digital Research Alliance cluster (tested target:
# Nibi). Run this on a LOGIN NODE — compute nodes have no outbound internet and
# this reaches both the CVMFS wheelhouse and PyPI.
#
#   cd <your path3d checkout>
#   bash hpc/setup_venv_nibi.sh
#
# VENV=... overrides where the venv goes. It defaults outside the checkout: the
# venv is ~400 MB of small files, it is not portable between clusters, and it
# has no business inside a git working tree.
set -euo pipefail

VENV="${VENV:-$HOME/venvs/path3d}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Everything except:
#   binaries      openslide-bin / pyvips-binary bundle libopenslide and libvips.
#                 The opencv/openslide/vips the cluster exposes through modules
#                 already cover this, and a second copy of a shared library in
#                 one process is how you get import-time symbol clashes. This is
#                 exactly why [binaries] is an extra and not a core dependency.
#   viz           napari + PyQt5: a long build for a GUI with no display here.
#   zeroreg3d     its opencv-python collides with [registration]'s headless cv2.
#   deeperhistreg not in the wheelhouse, and registration runs through VALIS.
# Add `niches` here once that branch lands on main.
EXTRAS="czi,registration,segmentation,nuclear_detection,training,volume,dev"

# ---------------------------------------------------------------- 1. modules
# shellcheck source=modules-nibi.sh
source "$REPO/hpc/modules-nibi.sh"
echo "python: $(command -v python)  ($(python -V 2>&1))"

# ------------------------------------------------------------------ 2. venv
# Built against the module python, without system site-packages. cv2 still
# resolves: the opencv module puts it on PYTHONPATH, not in site-packages.
if [[ -d "$VENV" ]]; then
    echo "!! $VENV already exists — including after a failed run, which leaves a" >&2
    echo "   half-built venv behind. Start clean:  rm -rf $VENV" >&2
    exit 1
fi
python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip setuptools wheel

# ------------------------------------------------------------- 3. install
# Versions are deliberately left to pip. The CVMFS wheelhouse is on find-links,
# so compiled packages come from there, and pure-Python ones pip fetches from
# PyPI. Pinning on top of that does not work: a freeze taken on one wheelhouse
# snapshot names compiled versions a later snapshot no longer serves (Nibi
# currently has pandas 2.2.1, not the 2.3.3 a sibling project froze), and PyPI
# cannot fill the gap for anything needing a compiled wheel. The constraints
# that actually matter — numpy<2, spatialdata<0.4, zarr<3, opencv<4.12 — are in
# pyproject.toml, where they are checked on every platform rather than one.
pip install -e "$REPO[$EXTRAS]"

# ------------------------------------------------------------- 4. verify
# A clean `pip install` proves nothing: the compiled extensions (pyvips,
# openslide, aicspylibczi, torch/CUDA) only fail once something imports them.
python - <<'PY'
import importlib, sys

MODS = [
    "numpy", "scipy", "pandas", "skimage", "tifffile", "PIL",
    "openslide", "pyvips", "cv2", "aicspylibczi",
    "torch", "torchvision", "torchstain", "timm", "huggingface_hub",
    "cellpose", "zarr", "spatialdata", "transformers",
    "valis", "albumentations", "segmentation_models_pytorch",
    "path3d",
]
bad = []
for m in MODS:
    try:
        importlib.import_module(m)
    except Exception as exc:                 # noqa: BLE001 - collect, don't raise
        bad.append(f"  {m}: {type(exc).__name__}: {exc}")

import numpy, torch
print(f"numpy {numpy.__version__}   (valis-wsi needs <2)")
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")

if bad:
    print("\nFAILED IMPORTS:\n" + "\n".join(bad), file=sys.stderr)
    sys.exit(1)
print("\nall imports OK")
PY

# cuda_available is False on a login node — they have no GPU. To check for real:
#   salloc --gpus=1 --mem=16G --time=0:15:00 --account=<your account>
#   source hpc/modules-nibi.sh && source "$VENV/bin/activate"
#   python -c "import torch; print(torch.cuda.get_device_name(0))"

echo
echo "venv: $VENV"
echo "activate with:  source $REPO/hpc/modules-nibi.sh && source $VENV/bin/activate"
