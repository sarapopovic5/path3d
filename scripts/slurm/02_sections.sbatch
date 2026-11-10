#!/bin/bash
#SBATCH --job-name=p3d_sections
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

# Stage 2 of 3, one array task: warp, segment and detect nuclei in this task's
# share of the sections (every N_TASKS-th, starting at the task index).
# Sections already on disk are skipped, so a failed task can be resubmitted
# alone: sbatch ... --array=<index> (see README).
# submit_all.sh sets --array, --gpus, --cpus-per-task and --mem, which override
# the defaults above (sized for one H100 3g.40gb MIG slice).
set -euo pipefail
source "$REPO/scripts/slurm/common.sh"

# Record which GPU or MIG slice this task got, for checking the smoke run.
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

p3d --stage sections --seg-model "$DPT_CKPT" \
    --task-index "$SLURM_ARRAY_TASK_ID" --n-tasks "$N_TASKS"
