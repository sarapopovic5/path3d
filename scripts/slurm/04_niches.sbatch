#!/bin/bash
#SBATCH --job-name=p3d_niches
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

# Niche stage, one array task: warp this task's share of the sections at
# native resolution (~0.22 um/px, what the niche model was trained at) and
# predict their tissue neighbourhoods. Needs only the register stage, so it
# runs alongside the sections array. Sections with predictions on disk are
# skipped and a half-done one resumes from its embedding checkpoint, so a
# timed-out task can be resubmitted alone: sbatch ... --array=<index>.
# submit_all.sh sets --array, --gpus, --cpus-per-task and --mem.
#
# UNI2-h must already be in the Hugging Face cache under $P3HOME: compute
# nodes have no internet. On a login node, after `hf auth login` inside the
# image with --home "$P3HOME":
#   python -c "from huggingface_hub import snapshot_download; snapshot_download('MahmoodLab/UNI2-h')"
set -euo pipefail
source "$REPO/scripts/slurm/common.sh"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

# Passed through apptainer's clean environment by the APPTAINERENV_ prefix.
export APPTAINERENV_HF_HUB_OFFLINE=1

args=(--stage niches --task-index "$SLURM_ARRAY_TASK_ID" --n-tasks "$NICHE_TASKS")
[[ -n ${NICHE_STEP_UM:-} ]] && args+=(--niche-step-um "$NICHE_STEP_UM")
[[ -n ${NICHE_VOXEL_UM:-} ]] && args+=(--niche-voxel-um "$NICHE_VOXEL_UM")
p3d "${args[@]}"
