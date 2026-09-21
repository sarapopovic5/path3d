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

On a cluster the native libraries come from environment modules and there is no
display, so install the bare package — no extras. On the Digital Research
Alliance of Canada clusters:

```bash
module load StdEnv/2023 gcc/12.3 cuda/12.2 python/3.10 opencv/4.10.0 java/11.0.22
virtualenv --no-download ~/venvs/path3d
source ~/venvs/path3d/bin/activate
pip install -e .
```

`opencv` comes from the module, so do not also install it through pip: two
copies of `cv2` will shadow each other. Run this on a login node, since compute
nodes have no outbound network. Record the result with `pip freeze` to make the
environment reproducible.

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
