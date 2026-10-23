#!/bin/bash
#SBATCH --job-name=p3d_sections
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Stage 2 of 3, one array task: warp, segment and detect nuclei in this task's
# share of the sections (every N_TASKS-th, starting at the task index).
# Sections already on disk are skipped, so a failed task can be resubmitted
# alone: sbatch ... --array=<index> (see README).
# submit_all.sh sets --array and --gpus.
set -euo pipefail
source "$REPO/scripts/slurm/common.sh"

p3d --stage sections --seg-model "$DPT_CKPT" \
    --task-index "$SLURM_ARRAY_TASK_ID" --n-tasks "$N_TASKS"
