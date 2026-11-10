#!/bin/bash
#SBATCH --job-name=p3d_volume
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# Stage 3 of 3: 3D volume and quantification report from the per-section
# files, once every array task has succeeded. CPU only.
set -euo pipefail
source "$REPO/scripts/slurm/common.sh"

p3d --stage volume
grep -h "QUANT.*FAILED" "$OUTDIR/logs/p3d_volume-$SLURM_JOB_ID.out" \
    && echo "^ quantification steps failed (the job still ends COMPLETED)" || true
