#!/bin/bash
#SBATCH --job-name=p3d_volume
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# Stage 3 of 3: 3D volume and quantification report from the per-section
# files, once every array task has succeeded. CPU only.
set -euo pipefail
source "$REPO/scripts/slurm/common.sh"

# A copy of this stage's output, for the failure check below. grep cannot read
# the job's own Slurm log while the job is writing to it ("input file is also
# the output"), so the check reads this copy instead. pipefail keeps p3d's
# exit status, so a crash still fails the job.
log="${SLURM_TMPDIR:-/tmp}/p3d_volume_$SLURM_JOB_ID.log"
p3d --stage volume 2>&1 | tee "$log"

if grep -E '^\[QUANT\] .* FAILED' "$log"; then
    echo "^ quantification steps failed (the job still ends COMPLETED; see report.md)"
fi
