# path3d

3D reconstruction of large tissues at cellular resolution from serially sectioned H&E whole slide images.

Based on the original [CODA](https://github.com/ashleylk/CODA) pipeline (Kiemen et al., *Nature Methods*, 2022), reimplemented in Python.

## Installation

```bash
pip install path3d
```

That is the whole pipeline — slide I/O, CZI support, registration, segmentation,
nuclear detection and volume assembly. Two things are left out, because each one
breaks an install target the other needs:

```bash
pip install "path3d[binaries]"   # bundled openslide + libvips, no system install
pip install "path3d[viz]"        # interactive 3D viewing (napari + Qt)
pip install "path3d[all]"        # both of the above
```

Registration requires a Java runtime (VALIS uses Bio-Formats).

`path3d` needs the OpenSlide and libvips native libraries. On a desktop the
`binaries` extra supplies both as pip wheels, so nothing has to be installed
system-wide. Where those libraries already exist, provide them instead of
installing the extra: it ships wheels only for mainstream platforms, and
elsewhere pip tries to compile them from source and fails.

`viz` is separate because it pulls in PyQt5 and the whole Qt5 runtime, which is
dead weight on a headless compute node.

### HPC clusters

On a cluster, run path3d from an Apptainer image instead of a venv. The image is
self-contained: Python 3.10, a JRE for VALIS, and the libvips and OpenSlide
builds from `[binaries]`. It covers preprocessing, registration, segmentation
and nuclear detection. Visualization is left out and runs locally.

The steps below follow the Alliance's
[Apptainer page](https://docs.alliancecan.ca/wiki/Apptainer) and are written for
[Nibi](https://docs.alliancecan.ca/wiki/Nibi), where every node, compute nodes
included, has internet access. Keep the checkout and your data under `/project`
or `/scratch`: those are the filesystems bound into the container, and `/home`
deliberately is not.

**Build** inside an interactive job, not on a login node. Apptainer's cache and
temp directories must be on non-networked disk, so they go in the job's local
`$SLURM_TMPDIR`. `APPTAINER_BIND=' '` stops host paths being mounted during the
build. Run this from the repo root:

```bash
salloc --account=<def-xxx> --time=1:00:00 --cpus-per-task=4 --mem=16G
module load apptainer
export APPTAINER_CACHEDIR=$SLURM_TMPDIR/apptainer/cache APPTAINER_TMPDIR=$SLURM_TMPDIR/apptainer/tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
APPTAINER_BIND=' ' apptainer build path3d.sif hpc/path3d.def
```

The build runs the image's self-test at the end. It reports
`cuda_available=False` because the job has no GPU, which is expected. To check
CUDA, run `apptainer test --nv path3d.sif` in a GPU job.

**Run** with the options the Alliance recommends. `-C` isolates the container
from the host environment. `-W $SLURM_TMPDIR` keeps temp files on local disk
rather than RAM. `-B` binds `/project` and `/scratch`, and `--nv` exposes the
GPU. `--home` points at a directory you create once, for example under your
group's `/project`. The pipeline's download caches live there and persist
between jobs: Hugging Face, Cellpose weights, and the Bio-Formats jars VALIS
fetches.

```bash
P3HOME=/project/<def-xxx>/$USER/path3d-home   # mkdir -p once
apptainer run -C --nv -W $SLURM_TMPDIR -B /project -B /scratch --home $P3HOME \
    path3d.sif scripts/run_full.py ...
```

UNI2-h is gated, so log in to Hugging Face once. The token is stored in
`$P3HOME`. Optionally, pre-fetch the models too, so that parallel jobs don't
each download them on first use:

```bash
A="apptainer run -C -W $SLURM_TMPDIR -B /project -B /scratch --home $P3HOME path3d.sif"
$A -c "from huggingface_hub import login; login()"
$A -c "from huggingface_hub import hf_hub_download as d; d('MahmoodLab/UNI2-h', 'pytorch_model.bin')"
$A -c "from cellpose import models; models.CellposeModel(gpu=False, pretrained_model='cpsam_v2')"
$A -c "from valis import registration as r; r.init_jvm(); r.kill_jvm()"
```

The image contains a snapshot of `src/`. To run a live checkout without
rebuilding, add `--env PYTHONPATH=$PWD/src`, run from the checkout root. Rebuild the image whenever the
dependencies change. `hpc/container-constraints.txt` holds the version pins,
and the image records its full resolved set in `/opt/path3d/pip-freeze.txt`.

### Development environment (conda)

`environment.yml` builds an env with every extra installed and a bundled JDK,
so VALIS registration works without a system Java:

```bash
conda env create -f environment.yml
conda activate path3d
```

Python dependencies are not duplicated in `environment.yml` — it installs the
project with `pip install -e .[all,dev]`, so `pyproject.toml` stays the single
source of truth.

## Supported formats

| Format | Extension | Backend |
|--------|-----------|---------|
| Aperio SVS | `.svs` | openslide |
| Hamamatsu NDPI | `.ndpi` | openslide |
| TIFF | `.tiff`, `.tif` | openslide |
| Leica SCN | `.scn` | openslide |
| 3DHISTECH MRXS | `.mrxs` | openslide |
| Zeiss CZI | `.czi` | aicspylibczi *(optional)* |


## License

MIT — see [LICENSE](LICENSE).
