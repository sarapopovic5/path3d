#!/bin/bash
# Build the whole-stack Xenium niche volume on the cluster.
#
# Runs `python -m path3d.niches.run` inside the path3d Apptainer image: niche
# prediction for every registered section (GPU), then the cubic volume and the
# rest of the run directory, laid out like full_volume_dpt/:
#
#   $OUT_DIR/volume_niches_${STEP_UM}um.zarr
#   $OUT_DIR/label_maps/NNNN_labels{,_rgb}.png
#   $OUT_DIR/quantification/{soft_volumetrics,per_section_profile}.csv
#   $OUT_DIR/manifest_thickness_8um.csv
#   $OUT_DIR/volume_build_metadata.json
#   $OUT_DIR/niches/{tiles,embeddings}/   (model output + resume checkpoints)
#
# For a multi-GPU run, use scripts/submit_niche_volume.sh, which submits this
# script as a job array (MODE=predict) plus a CPU job that builds the volume
# once the array is done (MODE=build).
#
# MODE=all (the default) does both in one job, e.g. in an salloc session:
#   SIF=/path/to/path3d.sif bash scripts/build_niche_volume.sh
# or as a single batch job -- the GPU is requested on the command line, not
# below, so the CPU-only build job does not hold one:
#   SIF=/path/to/path3d.sif sbatch --gres=gpu:h100:1 scripts/build_niche_volume.sh
#
# STEP_UM sets the spacing of tile centres, and so the voxel size (default 8:
# 8 um cubes, one per 8 um section). 32 is the model's own tile size and 16x
# cheaper. Preview first with STRIDE=4 -- every 4th tile per axis, 1/16 the
# cost, written to its own OUT_DIR.
#
# Resumable: finished sections are skipped and a half-done section resumes
# from its embedding checkpoint, so after a walltime kill just resubmit. The
# volume is built only once every section has a prediction.
#
# The repo is bind-mounted into the container and its src/ goes first on
# PYTHONPATH, so the code that runs is this checkout, not the copy baked into
# the image -- `git pull` takes effect without rebuilding the .sif.

#SBATCH --job-name=path3d-niches
#SBATCH --account=def-dcook
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/path3d_niches/logs/niches_%j.out
#SBATCH --error=/scratch/%u/path3d_niches/logs/niches_%j.err
# The log directory must exist BEFORE submitting -- SLURM opens these files
# before this script runs, and fails silently if it cannot:
#   mkdir -p $SCRATCH/path3d_niches/logs

set -euo pipefail

# ── Paths (override from the command line) ────────────────────────────────────
REPO_DIR="${SLURM_SUBMIT_DIR}"                                    # must be the repo root
SIF="${SIF:?set SIF=/path/to/path3d.sif}"
RUN_DIR="${RUN_DIR:-$SCRATCH/pycoda_full_new/output}"              # the registration run
REGISTERED="${REGISTERED:-$RUN_DIR/registered}"
# Section count and z-order come from here, not from globbing the directory.
MANIFEST="${MANIFEST:-$HOME/projects/def-dcook/active/pycoda/manifest_hpc.csv}"
MODE="${MODE:-all}"                    # all | predict (one array task) | build
STRIDE="${STRIDE:-1}"
STEP_UM="${STEP_UM:-8}"                # tile-centre spacing = voxel size
# Everything this job writes lives under $SCRATCH/path3d_niches:
#   niches_full_8um/              the full run at STEP_UM=8 (STRIDE=1)
#   preview_8um_stride4/          a STRIDE=4 preview, kept apart so it is never
#                                 mistaken for finished sections
#   logs/
# Runs with different steps go to different directories: a 32 um run's tiles
# are refused by an 8 um run rather than silently reused.
NICHES_ROOT="${NICHES_ROOT:-$SCRATCH/path3d_niches}"
if [[ "$STRIDE" == 1 ]]; then
    OUT_DIR="${OUT_DIR:-$NICHES_ROOT/niches_full_${STEP_UM}um}"
else
    OUT_DIR="${OUT_DIR:-$NICHES_ROOT/preview_${STEP_UM}um_stride${STRIDE}}"
fi
# Optional: the merged nuclear table from the DPT volume build
# (full_volume_dpt/nuclei_dpt_8um.parquet, built locally -- rsync it up).
# Attaches the same 9.5 M nuclei to the niche volume. Leave empty to skip.
NUCLEI_PARQUET="${NUCLEI_PARQUET:-}"

Z_SPACING_UM="${Z_SPACING_UM:-8}"      # spacing between consecutive manifest rows
VOXEL_UM="${VOXEL_UM:-$STEP_UM}"       # cube size; a whole multiple of Z_SPACING_UM
SMOOTH_UM="${SMOOTH_UM:-0}"            # in-plane smoothing sigma, microns (0 = off)

case "$MODE" in
    all|build) ;;
    predict)
        TASK_INDEX="${SLURM_ARRAY_TASK_ID:?MODE=predict must run as a job array task}"
        TASK_COUNT="${N_TASKS:-${SLURM_ARRAY_TASK_COUNT:?set N_TASKS}}"
        ;;
    *) echo "ERROR: MODE must be all, predict or build (got $MODE)" >&2; exit 1 ;;
esac

# Fail at submit time rather than after queueing.
[[ -f "$SIF" ]]              || { echo "ERROR: no image: $SIF" >&2; exit 1; }
[[ -d "$REGISTERED" ]]       || { echo "ERROR: no registered sections: $REGISTERED" >&2; exit 1; }
[[ -f "$MANIFEST" ]]         || { echo "ERROR: no manifest: $MANIFEST" >&2; exit 1; }
[[ -f "$REPO_DIR/src/path3d/niches/run.py" ]] \
    || { echo "ERROR: submit from the path3d repo root on the 3d-neighbourhoods branch" >&2; exit 1; }
if [[ -n "$NUCLEI_PARQUET" && ! -f "$NUCLEI_PARQUET" ]]; then
    echo "ERROR: no nuclei parquet: $NUCLEI_PARQUET" >&2; exit 1
fi

mkdir -p "$OUT_DIR" "$NICHES_ROOT/logs"

# ── Container environment ─────────────────────────────────────────────────────
module load apptainer 2>/dev/null || true

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
# Compute nodes have no internet: UNI2-h must already be in the Hugging Face
# cache under $HOME (it is, if the DPT segmentation ran from this account).
# To fetch it, on a login node after `hf auth login`:
#   apptainer exec $SIF python -c "from huggingface_hub import snapshot_download; snapshot_download('MahmoodLab/UNI2-h')"
export HF_HUB_OFFLINE=1
# The cluster points these at host CA bundles that do not exist inside the
# image. huggingface_hub builds an HTTP client even in offline mode, and it
# crashes on a missing SSL_CERT_FILE before it ever falls back to the cache.
# Unset, the container uses its own certificates.
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
export APPTAINERENV_PYTHONPATH="/opt/path3d-src/src"

# The cluster's Apptainer config already binds /project and /scratch (binding
# them again only prints warnings); /home holds the manifest symlinks and the
# Hugging Face cache. The repo is mounted at /opt/path3d-src.
# The host's module PYTHONPATH (cvmfs site-packages) is deliberately not
# forwarded -- APPTAINERENV_PYTHONPATH takes precedence, and Apptainer says so
# with a harmless warning.
GPU_FLAG=(--nv)
[[ "$MODE" == build ]] && GPU_FLAG=()     # the build step runs on CPU
CONTAINER=(
    apptainer exec ${GPU_FLAG[@]+"${GPU_FLAG[@]}"}
    --bind /home
    --bind "$TMPDIR"
    --bind "$REPO_DIR:/opt/path3d-src"
    "$SIF"
)

echo "======================================================"
echo "Job ID     : ${SLURM_JOB_ID}${SLURM_ARRAY_TASK_ID:+ (array task ${SLURM_ARRAY_TASK_ID})}"
echo "Mode       : ${MODE}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Image      : ${SIF}"
echo "Code       : ${REPO_DIR} ($(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?'))"
echo "Registered : ${REGISTERED}"
echo "Manifest   : ${MANIFEST}"
echo "Output     : ${OUT_DIR}"
echo "Grid       : tiles every ${STEP_UM} um, ${VOXEL_UM} um cubes from ${Z_SPACING_UM} um sections, stride ${STRIDE}"
echo "Start      : $(date)"
echo "======================================================"

# Pre-flight: the image has the niche dependencies, the code resolves to this
# checkout, and -- when predicting -- the GPU is visible and UNI2-h is cached,
# before an hour in queue turns into a crash on the first import.
"${CONTAINER[@]}" env PREFLIGHT_MODE="$MODE" python - <<'PY'
import importlib.util, os, sys
missing = [m for m in ("sklearn", "joblib", "spatialdata", "anndata", "pyarrow",
                       "zarr", "timm", "huggingface_hub", "tifffile", "PIL")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"image is missing {missing}: rebuild it with path3d[niches,volume]")
import path3d, torch
from huggingface_hub import snapshot_download
print(f"path3d from : {path3d.__file__}")
if not path3d.__file__.startswith("/opt/path3d-src/"):
    sys.exit("path3d is not being imported from the bind-mounted checkout")
if os.environ["PREFLIGHT_MODE"] == "build":
    sys.exit(0)  # building needs neither the GPU nor the encoder
print(f"CUDA        : {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("no GPU visible inside the container (is --nv set?)")
try:
    print(f"UNI2-h      : {snapshot_download('MahmoodLab/UNI2-h', local_files_only=True)}")
except Exception as exc:
    sys.exit(f"UNI2-h is not in the Hugging Face cache ({exc}). Download it on "
             f"a login node -- see the comment above HF_HUB_OFFLINE in this script.")
# The same lookup timm.create_model does. Unlike local_files_only above, it
# goes through huggingface_hub's HTTP client, so it catches environment
# problems (e.g. a host SSL_CERT_FILE) that only surface when the model loads.
from timm.models._hub import download_from_hf
try:
    download_from_hf("MahmoodLab/UNI2-h", "config.json")
except Exception as exc:
    sys.exit(f"timm cannot resolve UNI2-h offline: {exc!r}")
PY

ARGS=(
    --manifest "$MANIFEST"
    --registered "$REGISTERED"
    --out "$OUT_DIR"
    --z-spacing-um "$Z_SPACING_UM"
    --step-um "$STEP_UM"
    --voxel-um "$VOXEL_UM"
    --smooth-um "$SMOOTH_UM"
    --stride "$STRIDE"
)
case "$MODE" in
    all)     ARGS+=(--device cuda) ;;
    predict) ARGS+=(--device cuda --predict-only
                    --task-index "$TASK_INDEX" --task-count "$TASK_COUNT") ;;
    build)   ARGS+=(--skip-predict) ;;
esac
[[ "$MODE" != predict && -n "$NUCLEI_PARQUET" ]] && ARGS+=(--nuclei-parquet "$NUCLEI_PARQUET")

srun "${CONTAINER[@]}" python -m path3d.niches.run "${ARGS[@]}"

echo "======================================================"
echo "Done   : $(date)"
if [[ "$MODE" != predict ]]; then
    echo "Volume : ${OUT_DIR}/volume_niches_${VOXEL_UM}um.zarr"
fi
echo "Run dir: ${OUT_DIR}"
echo "======================================================"
