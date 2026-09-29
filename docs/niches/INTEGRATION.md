# Building a 3D neighbourhood volume with `path3d.niches`

`docs/niches/MODEL_CARD.md` is the Cook Lab's model card, unmodified — it is the
authority on what the classifier can and cannot do. This document covers only
the part path3d adds: getting from a stack of CZI sections to one 3D volume,
and the decisions that changes.

The model card names the gap this fills:

> Tile coordinates are in each slide's own level-0 pixels, so **sections are not
> co-registered to one another** — align them yourself.

path3d is that alignment step.

## The pipeline

```
CZI sections
  ↓  slide_io.create_manifest           manifest.csv — row order IS z order
  ↓  pipeline.run_pipeline              VALIS registration → registered/NNNN.ome.tiff
  ↓  niches.predict_sections            per-section tiles_niches.csv on a 32 µm grid
  ↓  niches.load_sections               rasterise onto the shared canvas
  ↓  niches.build_niche_volume          SpatialData: probabilities + labels
  ↓  visualization.view_volume(tissue_type="HGSC_niches")
```

```python
from path3d.niches import run_niche_pipeline

sdata = run_niche_pipeline(
    "manifest.csv",
    "run_out/registered",
    "run_out/niches",
    z_spacing_um=12.0,   # spacing between CONSECUTIVE MANIFEST ROWS
    smooth_um=100.0,
)
```

Per-section CSVs are written as each section finishes, so a walltime kill loses
only the section in flight; re-running skips what is already on disk.

### The whole stack, as one run directory

`python -m path3d.niches.run` does the same and writes the result in the layout
of a `run_pipeline` + volume-build run (`full_volume_dpt/`):

```
OUT/volume_niches_32um.zarr       labels + probabilities (+ nuclei)
OUT/label_maps/NNNN_labels{,_rgb}.png
OUT/quantification/soft_volumetrics.csv, per_section_profile.csv
OUT/manifest_thickness_8um.csv
OUT/volume_build_metadata.json
OUT/niches/{tiles,embeddings}/
```

On the cluster, from the repo root, inside the path3d Apptainer image:

```bash
mkdir -p $SCRATCH/path3d_niches/logs   # outputs land in $SCRATCH/path3d_niches too
SIF=/path/to/path3d.sif STRIDE=4 sbatch scripts/build_niche_volume.sh   # preview
SIF=/path/to/path3d.sif sbatch scripts/build_niche_volume.sh            # full run
```

The job bind-mounts the checkout over the image's copy of path3d, so a
`git pull` takes effect without rebuilding the image. Resubmit it after a
walltime kill. Everything after prediction runs from `niches/tiles` alone, so
`--skip-predict` rebuilds the volume without a GPU.
`--nuclei-parquet full_volume_dpt/nuclei_dpt_8um.parquet` attaches the same
nuclei the DPT volume carries. View the result with
`python scripts/view_volume.py OUT/volume_niches_32um.zarr`; it detects a niche
volume and uses the `HGSC_niches` palette.

## Run the model on the registered sections, not the raw slides

Two hard reasons:

1. `predict_niches.py` reads through `tifffile`/`zarr` and **cannot open a CZI at
   all**. `registration.warp_and_save_section` has a purpose-built CZI adapter
   (`registration._CziReaderAdapter`) that exists because Bio-Formats
   *segfaults* on JPEGXR-compressed Zeiss CZIs.
2. Every section warped from one VALIS run lands on the same aligned canvas.
   That is what makes the per-section grids stackable —
   `build_niche_volume` does `np.stack`, so the grids must be identically
   shaped, and they are only identically shaped if they were derived from a
   shared canvas.

The grid is always computed from the canvas dimensions, never from the tiles
that happened to land on tissue. A section with a sliver of tissue and a section
that is wall-to-wall tumour still produce the same grid shape; the difference
shows up as NaN, not as a different array size. `build_niche_volume` re-checks
this and fails loudly rather than producing a sheared volume.

## Resolution: warp at the slide's native mpp, not `SEG_MPP`

**This is the one that silently costs you accuracy.**

The model crops a 112 µm window and resizes it to 224×224. It was trained at
0.2201 µm/px, where that window is **509 px** and is downsampled 2.3× into the
network. path3d's default `cfg.SEG_MPP` is 0.5 µm/px, where the same window is
**224 px** — the right physical size, but carrying roughly a fifth of the detail,
with no resize happening at all.

So pass `target_mpp=0.2201` (or whatever `CziReader.get_mpp()` reports) to
`warp_and_save_section` for the niche pass. Check your own value — the model
card is blunt that theirs need not be yours, and `_source_level_for_mpp` raises
rather than upsample if you ask for finer than level 0.

One warp serves both consumers: the output is pyramidal, so
`segmentation.predict_section` calling `best_level_for_mpp(0.5)` picks a
~0.44 µm/px level off the same file. You do not need to warp twice.

`tests/test_niches_ome_tiff.py` reports this as an `xfail` against whatever
stack you point it at, naming the actual ratio.

## `--mpp` is now read from the file

The model card warns that a wrong `--mpp` "silently changes the physical size of
every tile and degrades everything". `predict.read_mpp` reads it from the
image's own OME-XML `PhysicalSizeX`, falling back to the TIFF resolution tags —
both of which `warp_and_save_section` stamps. That is strictly safer than a
human retyping a number. Pass `mpp=` only to override metadata you know is
wrong.

## The z-pitch trap

`volume.build_volume` sets the label volume's z spacing to `target_voxel_um`
**unconditionally** and ignores `thickness_um`. If you rasterise to a 32 µm grid
and set `target_voxel_um=32.0` while your sections are 12 µm apart, the volume
comes out **stretched 2.7× in z** and every volume and surface area you compute
from it is wrong.

`build_niche_volume` does not inherit that. Voxels are anisotropic by default —
`(z_spacing_um, grid_um, grid_um)` — and the anisotropy is encoded in the affine
where it belongs. In-plane stays 32 µm because that is genuinely the data's
resolution; upsampling it to 12 µm costs 7× the memory and invents nothing.

`z_spacing_um` is the spacing between **consecutive manifest rows** — section
thickness × sampling interval. If you cut 4 µm sections and imaged every third,
it is 12 µm, not 4 µm. Omit it and it is read from a uniform `thickness_um`
column; it raises rather than guess.

Pass `target_voxel_um` for cubic voxels. You need it for
`quantification.compute_volumetrics`, which takes a single scalar `voxel_um` and
assumes cubic voxels. It must be a whole multiple of the section spacing, and
that many consecutive sections are **averaged into one slab**: 32 µm cubes from
8 µm sections average 4. That is the z counterpart of rasterising at a coarser
`grid_um`. A slab voxel is tissue when at least half its sections are, the same
majority rule the in-plane grid applies, so the tissue boundary does not move
with the cube size. Trailing sections that do not fill a whole slab (149 = 37 × 4
+ 1) are left out, with a warning, and listed in
`volume_build_metadata.json`. A cube finer than the section spacing is refused,
since it would have to invent data between sections.

## Probabilities are the deliverable; the label volume is for viewing

The model card:

> Use the continuous `p_*` columns wherever you can, and the per-slide calls
> when you need a binary mask.

A global argmax under-calls rare classes even when ranking is excellent — on
their 4.5 %-epithelium tumour, AUC 0.935 against F1 0.23. A label volume built
by argmax inherits that bias, so if you report "immune occupies X % of this
tumour" off the labels, the bias is in your headline number.

`build_niche_volume` therefore produces both:

| element | model | use |
|---|---|---|
| `images["niche_probabilities"]` | `(c, z, y, x)` float32, NaN off-tissue | anything you report |
| `labels["tissue_labels"]` | `(z, y, x)` uint8 argmax | `view_volume`, `compute_volumetrics` |

The label element is named `tissue_labels` so `visualization.view_volume` and
`quantification` find it with no arguments. Pass `tissue_type="HGSC_niches"` to
get the right palette and class names.

## Smoothing instead of a 250 µm grid

The model card recommends reporting at 250 µm because per-tile agreement between
serial sections was "substantially weaker than per-region". Because the
probabilities are kept, you can have that noise reduction without the coarse
grid: `smooth_um` applies a NaN-aware in-plane Gaussian before the argmax.
`smooth_um=100.0` is a reasonable starting point.

`smooth_z_sigma` smooths across sections too, and is **off by default** — it
mixes neighbouring sections' probabilities, so it is only justified if you trust
the registration to better than one grid cell. Neither smoother invents tissue:
a voxel with no prediction stays background.

To reproduce the model card's reporting resolution exactly, rasterise with
`grid_um=250.0` instead — that averages the tiles in each cell, which is the
correct way to coarsen.

## Cost, caching and previews

The model card quotes ~17 tiles/s on an M3 Max. Measured on a real 25 mm²
section here it was **~5 tiles/s**, so budget from your own throughput, not the
quoted figure: 25 000 tiles took ~80 min for one section. A 150-section stack is
a multi-day GPU job.

The embedding pass is the *only* expensive part — the classifier on top runs in
milliseconds. So embeddings are checkpointed to disk every 2048 tiles
(`output_dir/embeddings/NNNN.npz`, ~75 MB per 25 000 tiles) and reloaded on a
re-run. That buys two things:

- **An interrupted section resumes** instead of restarting. Without it, a
  walltime kill at tile 24 000 of 24 598 discards the whole section.
- **Re-scoring is free.** Change `--quantile`, or swap in a retrained
  classifier, without paying for UNI2-h again:

  ```python
  from path3d.niches import load_embedding_cache, load_model
  from path3d.niches.predict import classify_embeddings

  emb, done, _ = load_embedding_cache("out/embeddings.npz")
  p, cols = classify_embeddings(emb[done], load_model(), quantile=0.9)
  ```

A cache is keyed by a fingerprint of the tile coordinates, `mpp` and window
size. If any of those change it is discarded whole and recomputed — never
partially reused, which would silently mix embeddings from two different tile
sets. Pass `--no-cache` / `cache_embeddings=False` to disable.

**For a quick look, use `--stride`.** It keeps every Nth grid cell in each axis,
so `--stride 4` covers the whole section at 1/16 the cost. The kept tiles stay
on their original grid positions, so a preview rasterises into exactly the same
cells a full run would — just sparser, with the gaps left NaN. Prefer it over
`--limit`, which takes the first N tiles in block order and therefore samples
one corner of the section.

```bash
python -m path3d.niches.predict SECTION.ome.tiff --out preview --stride 4
```

## Before you trust a batch

The model card's own caveat applies with force here: that script had never been
run on a real slide from a different scanner. Run one section, look at
`niche_maps.png`, and check the epithelium map against the H&E before queueing
150 of them. And note the model is **HGSC only**, post-Xenium H&E, one scanner —
anything else is extrapolation.

## Tests

```bash
pytest tests/                                    # no data needed; real-image tests skip
pytest tests/ --ome-tiff /path/to/0000.ome.tiff  # + one registered section
pytest tests/ --ome-tiff /path/to/0000.ome.tiff --run-encoder   # + the gated UNI2-h
```

The path is supplied at runtime and never committed — path3d is a public
repository and the slides are patient-derived. `PATH3D_TEST_OME_TIFF` works too.
