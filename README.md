# path3d

3D reconstruction of large tissues at cellular resolution from serially sectioned H&E whole slide images.

Based on the original [CODA](https://github.com/ashleylk/CODA) pipeline (Kiemen et al., *Nature Methods*, 2022), reimplemented in Python.

## Installation

```bash
pip install path3d
```

Optional extras:

```bash
pip install "path3d[czi]"            # Zeiss CZI support
pip install "path3d[registration]"   # serial section registration (VALIS)
pip install "path3d[viz]"            # interactive 3D viewing (napari)
pip install "path3d[niches]"         # 3D Xenium tissue neighbourhoods from H&E
pip install "path3d[all]"            # everything
```

The `registration` extra requires a Java runtime (VALIS uses Bio-Formats).
`openslide` and `libvips` are bundled as pip wheels — no system install needed.

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

## Conventions

- Coordinates are 0-based; 2D is `(row, col)` = `(y, x)`, 3D is `(z, row, col)`
- Physical units are microns
- `read_region` always returns `(H, W, 3)` uint8 RGB — alpha is stripped

## 3D tissue neighbourhoods

`path3d.niches` predicts Xenium-derived tissue neighbourhoods
(`epithelium | immune | stroma | acellular`) from H&E alone and stacks the
per-section maps into a 3D volume, using path3d's registration to co-register
the sections:

```python
from path3d.niches import run_niche_pipeline

sdata = run_niche_pipeline(
    "manifest.csv",
    "run_out/registered",   # from pipeline.run_pipeline
    "run_out/niches",
    z_spacing_um=12.0,      # spacing between CONSECUTIVE manifest rows
    smooth_um=100.0,
)
```

You get a `(c, z, y, x)` probability volume and a `(z, y, x)` argmax label
volume. Prefer the probabilities for anything you report — a global argmax
under-calls rare classes.

The classifier and its stain normaliser are vendored from the Cook Lab
`2026_final_xenium_analysis` package.
[`docs/niches/MODEL_CARD.md`](docs/niches/MODEL_CARD.md) is their model card,
unmodified, and is the authority on validation and limits — **HGSC only**, one
scanner. [`docs/niches/INTEGRATION.md`](docs/niches/INTEGRATION.md) covers the
resolution, z-pitch and argmax decisions the 3D path adds.

Requires Hugging Face access to the gated
[`MahmoodLab/UNI2-h`](https://huggingface.co/MahmoodLab/UNI2-h) (request access,
then `hf auth login`).

## Documentation

See [`docs/`](docs/) for the pipeline overview, algorithm notes, and references.

## Tests

```bash
pytest tests/                                    # no data needed
pytest tests/ --ome-tiff /path/to/0000.ome.tiff  # + one registered section
```

Tests that need real slides take the path at runtime and skip without it — no
data location is ever committed.

## License

MIT — see [LICENSE](LICENSE).
