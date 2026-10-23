#!/bin/bash
# Submit a staged run as three chained Slurm jobs:
#
#   1. register  CPU        VALIS over the whole stack
#   2. sections  GPU array  N_TASKS tasks, each warping, segmenting and
#                           detecting nuclei in its share of the sections
#   3. volume    CPU        3D volume + quantification
#
# Each job starts only when the one before succeeded (afterok; for the array,
# when every task did). Logs go to $OUTDIR/logs/.
#
#   bash scripts/slurm/submit_all.sh local/hpc.env            # whole run
#   bash scripts/slurm/submit_all.sh local/hpc.env sections   # from stage 2
#   bash scripts/slurm/submit_all.sh local/hpc.env volume     # stage 3 only
#
# Starting from a later stage reuses what is already in $OUTDIR: rerunning
# "sections" after a failed task redoes only the sections that never finished.
set -euo pipefail

env_file=${1:?usage: submit_all.sh <env file> [register|sections|volume]}
from=${2:-register}
set -a
source "$env_file"
set +a

for v in ACCOUNT REPO SIF P3HOME OUTDIR; do
    [[ -n ${!v:-} ]] || { echo "$v is not set in $env_file" >&2; exit 1; }
done
if [[ ! $from =~ ^(register|sections|volume)$ ]]; then
    echo "unknown stage: $from" >&2; exit 1
fi
if [[ $from == register && ! -f ${MANIFEST:-} ]]; then
    echo "manifest not found: ${MANIFEST:-}" >&2; exit 1
fi
if [[ $from != volume && ! -f ${DPT_CKPT:-} ]]; then
    echo "checkpoint not found: ${DPT_CKPT:-}" >&2; exit 1
fi
export N_TASKS=${N_TASKS:-10}
GPU=${GPU:-h100:1}

slurm_dir="$REPO/scripts/slurm"
mkdir -p "$OUTDIR/logs"
common=(--parsable --account="$ACCOUNT" --export=ALL --chdir="$REPO")
after=()

if [[ $from == register ]]; then
    j1=$(sbatch "${common[@]}" --output="$OUTDIR/logs/%x-%j.out" "$slurm_dir/01_register.sbatch")
    echo "register  $j1"
    after=(--dependency=afterok:$j1)
fi
if [[ $from != volume ]]; then
    j2=$(sbatch "${common[@]}" ${after[@]+"${after[@]}"} --output="$OUTDIR/logs/%x-%A_%a.out" \
        --array=0-$((N_TASKS - 1)) --gpus="$GPU" "$slurm_dir/02_sections.sbatch")
    echo "sections  $j2 ($N_TASKS tasks, $GPU each)"
    after=(--dependency=afterok:$j2)
fi
j3=$(sbatch "${common[@]}" ${after[@]+"${after[@]}"} --output="$OUTDIR/logs/%x-%j.out" "$slurm_dir/03_volume.sbatch")
echo "volume    $j3"
echo "logs:     $OUTDIR/logs/"
