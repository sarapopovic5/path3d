# Sourced by the stage scripts. p3d runs scripts/run_full.py inside the image
# on the checkout's live src/, with the options the README recommends.
# REPO, SIF, P3HOME and OUTDIR come from the env file via submit_all.sh.
#
# -C clears the environment inside the container, so everything run_full.py
# needs is passed as a flag, not through env vars.

p3d() {
    module load apptainer
    cd "$REPO"
    local nv=()
    [[ -n ${CUDA_VISIBLE_DEVICES:-} ]] && nv=(--nv)
    apptainer run -C ${nv[@]+"${nv[@]}"} -W "${SLURM_TMPDIR:-/tmp}" -B /project -B /scratch \
        --home "$P3HOME" --pwd "$(pwd -P)" --env PYTHONPATH="$(pwd -P)/src" \
        "$SIF" scripts/run_full.py --out "$OUTDIR" "$@" ${EXTRA_ARGS:-}
}
