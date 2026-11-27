#!/bin/bash
#SBATCH --job-name=p3d_niches_volume
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# Niche volume, once every niches task has succeeded (and the volume stage,
# when it ran, so its nuclei parquet is attached). Writes
# $OUTDIR/neighbourhoods/volume_niches_<v>um.zarr, per-section label maps and
# quantification/soft_volumetrics.csv. CPU only.
set -euo pipefail
source "$REPO/scripts/slurm/common.sh"

args=(--stage niches-volume)
[[ -n ${NICHE_VOXEL_UM:-} ]] && args+=(--niche-voxel-um "$NICHE_VOXEL_UM")
[[ -n ${NICHE_SMOOTH_UM:-} ]] && args+=(--niche-smooth-um "$NICHE_SMOOTH_UM")
p3d "${args[@]}"
