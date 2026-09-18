# Source this BEFORE activating the venv, in exactly this order, both when you
# build the environment and in every job script that uses it. The venv was built
# against these modules; loading a different set (or none) breaks the compiled
# extensions at import time, not at install time.
#
#   source <checkout>/hpc/modules-nibi.sh
#   source ~/venvs/path3d/bin/activate
#
# This is the module set that the sibling pycoda environment ran under, copied
# unchanged because it is the part that was already proven on this cluster. Do
# not "upgrade" it casually — python/3.10 is what pyproject's requires-python
# and the napari pins assume, and cuda/12.2 is what the pinned torch expects.

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.10
module load opencv/4.10.0
module load java/11.0.22
# pyarrow ([volume]) is not a real wheel here: the wheelhouse lists it at the
# sentinel version 9999, meaning "provided by a module, do not pip install".
module load arrow

# VALIS shells out to Bio-Formats through scyjava/jgo, which reads JAVA_HOME.
# The java module sets it; assert rather than discover a silent failure later.
: "${JAVA_HOME:?java module did not set JAVA_HOME}"

# scyjava/jgo resolve Maven artifacts over the network on first use, and
# compute nodes have no outbound internet. Keep the caches on a shared
# filesystem so the login-node warm-up is visible from the job.
export M2_HOME="${M2_HOME:-$HOME/.m2}"
export JGO_CACHE_DIR="${JGO_CACHE_DIR:-$HOME/.jgo}"

# Hugging Face caches model weights (UNI2-h, used by the niche classifier, is
# ~2.5 GB and gated). $HOME is quota-tight, so default the cache next to the
# checkout rather than into ~/.cache, and pre-download on a login node:
#     hf auth login
#     python -c "import timm; timm.create_model('hf-hub:MahmoodLab/UNI2-h', pretrained=True)"
# Then set HF_HUB_OFFLINE=1 in job scripts, so a cache miss fails loudly instead
# of hanging on a network call a compute node can never complete.
export HF_HOME="${HF_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/hf_cache}"
