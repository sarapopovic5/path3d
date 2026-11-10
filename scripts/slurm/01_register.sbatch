#!/bin/bash
#SBATCH --job-name=p3d_register
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

# Stage 1 of 3: VALIS over the whole stack, CPU only. Writes the registration
# and volume_build_metadata.json, which stages 2 and 3 read back.
# Submitted by submit_all.sh, which sets the account, log path and env.
set -euo pipefail
source "$REPO/scripts/slurm/common.sh"

spacing=()
if [[ -n ${SECTION_THICKNESS_UM:-} ]]; then
    spacing=(--section-thickness-um "$SECTION_THICKNESS_UM" --sampling-interval "$SAMPLING_INTERVAL")
fi

p3d --stage register --manifest "$MANIFEST" ${spacing[@]+"${spacing[@]}"}
