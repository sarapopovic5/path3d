#!/bin/bash
# Submit the whole-stack niche volume as a GPU job array plus one build job.
#
#   SIF=/path/to/path3d.sif bash scripts/submit_niche_volume.sh
#
# Submits scripts/build_niche_volume.sh twice:
#   1. MODE=predict as an N_TASKS-way job array. Task i predicts manifest rows
#      i, i+N, i+2N, ... (interleaved, so every task gets a similar mix of big
#      and small sections). One H100 each.
#   2. MODE=build, CPU only, starting after the whole array has ended
#      (afterany). It assembles the volume from niches/tiles and fails,
#      naming the sections, if any task did not finish.
#
# At the default 8 um step a section is ~400 000 tiles, about an hour on an
# H100, so 149 sections over 10 tasks is ~15 h per task. If a task hits the
# walltime, resubmit just that task and the build -- finished sections are
# skipped, and a half-done one resumes from its embedding checkpoint:
#   SIF=... TASKS=3,7 bash scripts/submit_niche_volume.sh
#
# Every setting of build_niche_volume.sh (STEP_UM, STRIDE, OUT_DIR,
# NUCLEI_PARQUET, ...) passes through from the environment.

set -euo pipefail

cd "$(dirname "$0")/.."                      # repo root: jobs bind-mount it
: "${SIF:?set SIF=/path/to/path3d.sif}"
N_TASKS="${N_TASKS:-10}"
TASKS="${TASKS:-0-$((N_TASKS - 1))}"
NICHES_ROOT="${NICHES_ROOT:-$SCRATCH/path3d_niches}"
LOGS="$NICHES_ROOT/logs"
mkdir -p "$LOGS"
export SIF N_TASKS NICHES_ROOT

predict=$(sbatch --parsable \
    --array="$TASKS" \
    --job-name=path3d-niches-predict \
    --gres=gpu:h100:1 --mem=64G --time=24:00:00 \
    --output="$LOGS/niches_predict_%A_%a.out" \
    --error="$LOGS/niches_predict_%A_%a.err" \
    --export=ALL,MODE=predict \
    scripts/build_niche_volume.sh)

build=$(sbatch --parsable \
    --dependency="afterany:${predict}" \
    --job-name=path3d-niches-build \
    --cpus-per-task=4 --mem=96G --time=4:00:00 \
    --output="$LOGS/niches_build_%j.out" \
    --error="$LOGS/niches_build_%j.err" \
    --export=ALL,MODE=build \
    scripts/build_niche_volume.sh)

echo "predict array : ${predict}  (tasks ${TASKS} of ${N_TASKS})"
echo "build         : ${build}  (runs after the array ends)"
echo "logs          : ${LOGS}"
echo "watch         : squeue -u $USER"
