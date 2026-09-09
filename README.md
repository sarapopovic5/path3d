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
pip install "path3d[all]"            # everything
```

The `registration` extra requires a Java runtime (VALIS uses Bio-Formats).
`openslide` and `libvips` are bundled as pip wheels — no system install needed.

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

## Documentation

See [`docs/`](docs/) for the pipeline overview, algorithm notes, and references.

## License

MIT — see [LICENSE](LICENSE).
