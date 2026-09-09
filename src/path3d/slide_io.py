"""WSI reading, metadata extraction, tile-based image reading

Two backends share a common SlideReader interface:
  - OpenSlideReader: SVS, NDPI, TIFF, SCN, MRXS, and any other format supported by openslide-python
  - CziReader: Zeiss CZI files, via aicspylibczi (wraps libCZI).

Use open_slide(path) to get the right reader automatically.  The rest of thevpipeline only ever calls SlideReader methods

Memory safety: never loads a full WSI into RAM.  All pixel access
goes through read_region() which decodes only the requested tiles from disk
"""

from __future__ import annotations

import csv
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

import numpy as np


# Abstract interface

class SlideReader(ABC):
    """Common interface for WSI backends

    Coordinate conventions (matches OpenSlide throughout):
      - level 0     = full resolution
      - location    = (x, y) in level-0 pixel coordinates, 0-based
      - size        = (width, height) at the requested level
      - (row, col)  = (y, x) — NumPy / scikit-image order elsewhere in path3d
    """

    # Required properties 

    @property
    @abstractmethod
    def dimensions(self) -> tuple[int, int]:
        """(width, height) at level 0."""

    @property
    @abstractmethod
    def level_count(self) -> int:
        """Number of available pyramid levels."""

    @property
    @abstractmethod
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        """(width, height) at each pyramid level."""

    @property
    @abstractmethod
    def level_downsamples(self) -> tuple[float, ...]:
        """Downsample factor relative to level 0 for each level."""

    # Required methods 

    @abstractmethod
    def get_mpp(self) -> float:
        """Microns per pixel at level 0.

        Raises ValueError if calibration metadata is absent.
        """

    @abstractmethod 
    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Largest pyramid level whose downsample does not exceed *downsample*."""

    @abstractmethod
    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        """Read a rectangular region and return an (H, W, 3) uint8 RGB array.

        Args:
            location: (x, y) top-left corner in level-0 pixel coordinates.
            level:    Pyramid level to read from.
            size:     (width, height) of the region *at the requested level*.
        """

    @abstractmethod
    def get_thumbnail(self, max_size: int = 2000) -> np.ndarray:
        """Return a downsampled (H, W, 3) uint8 RGB thumbnail.

        The longer dimension is scaled to max_size, aspect ratio is preserved
        """

    @abstractmethod
    def close(self) -> None:
        """Release any open file handles or resources"""

    # Context manager 
    def __enter__(self) -> SlideReader:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # Derived helpers 

    @staticmethod
    def _fill_unscanned(img: np.ndarray) -> np.ndarray:
        """Replace fully-black pixels (unscanned tiles) with white in-place"""
        img[np.all(img == 0, axis=-1)] = 255
        return img

    def get_mpp_at_level(self, level: int) -> float:
        
        return self.get_mpp() * self.level_downsamples[level]

    def best_level_for_mpp(self, target_mpp: float) -> int:
        """Pyramid level whose resolution is closest to target mpp

        Equivalent to OpenSlide's get_best_level_for_downsample but in physical units
        """
        mpp0 = self.get_mpp()
        return self.get_best_level_for_downsample(target_mpp / mpp0)

    def iter_tiles(
        self,
        pyramid_level: int,
        tile_size: int = 1024,
        overlap: int = 0,
    ) -> Generator[tuple[tuple[int, int], tuple[int, int], np.ndarray], None, None]:
        """Yield tiles covering the full slide at *pyramid_level* in row-major order.

        Edge tiles are clipped to the slide boundary and may be smaller than
        tile_size.  For segmentation use overlap=128 or 256 and discard the
        outer frame of predictions on each tile.

        Yields:
            (col_row, actual_size, tile_rgb) where
                col_row - (col, row) 0-based tile index
                actual_size — (width, height) of this tile in level pixels
                tile_rgb  — (H, W, 3) uint8 RGB array
        """
        level_w, level_h = self.level_dimensions[pyramid_level]
        downsample = self.level_downsamples[pyramid_level]
        stride = tile_size

        row_idx = 0
        y_level = 0
        while y_level < level_h:
            col_idx = 0
            x_level = 0
            while x_level < level_w:
                x0 = max(0, x_level - overlap)
                y0 = max(0, y_level - overlap)
                x1 = min(level_w, x_level + tile_size + overlap)
                y1 = min(level_h, y_level + tile_size + overlap)
                w, h = x1 - x0, y1 - y0

                # Convert level-space coords to level-0 for read_region
                x0_l0 = int(x0 * downsample)
                y0_l0 = int(y0 * downsample)

                tile = self.read_region((x0_l0, y0_l0), pyramid_level, (w, h))
                yield (col_idx, row_idx), (w, h), tile

                col_idx += 1
                x_level += stride
            row_idx += 1
            y_level += stride



# OpenSlide backend. Exists to wrap openslide in SlideReader interface, 
# so it can be used interchangeably with CziReader. 


class OpenSlideReader(SlideReader):
    """SlideReader backed by openslide-python.

    Supports SVS, NDPI, TIFF, SCN, MRXS, BIF, and any other format that the
    installed openslide library handles.  Does not support CZI. use CziReader for Zeiss files
    """

    def __init__(self, path: str | Path) -> None:
        try:
            import openslide
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "openslide-python and the OpenSlide C library are required to read this format. "
                "Install with: pip install openslide-python openslide-bin"
            ) from exc
        self._slide = openslide.OpenSlide(str(path))

    @property
    def dimensions(self) -> tuple[int, int]:
        return self._slide.dimensions

    @property
    def level_count(self) -> int:
        return self._slide.level_count

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return self._slide.level_dimensions

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return self._slide.level_downsamples

    def get_mpp(self) -> float:
        val = self._slide.properties.get("openslide.mpp-x") 
        if val is not None:
            return float(val) 
        raise ValueError(
            "Cannot determine MPP from slide metadata. "
        )

    def get_best_level_for_downsample(self, downsample: float) -> int:
        return self._slide.get_best_level_for_downsample(downsample)

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        # read_region returns RGBA — always strip alpha channel
        region = self._slide.read_region(location, level, size).convert("RGB")
        return self._fill_unscanned(np.asarray(region).copy())

    def get_thumbnail(self, max_size: int = 2000) -> np.ndarray:
        w, h = self.dimensions
        scale = max_size / max(w, h) # preserves aspect ratio
        thumb = self._slide.get_thumbnail((int(w * scale), int(h * scale))).convert("RGB")
        return self._fill_unscanned(np.asarray(thumb).copy())

    def close(self) -> None:
        self._slide.close()



# CZI backend

class CziReader(SlideReader):
    """SlideReader backed by aicspylibczi (wraps the official Zeiss libCZI).

    Supports Zeiss CZI files, including large mosaic (tiled) whole-slide scans.

    Pyramid levels are simulated; CZI does not store a fixed set of levels like
    SVS/NDPI, but libCZI decodes at any scale.
    This class pre-defines downsample factors [1, 2, 4, 8, 16, 32, 64] and
    exposes them as levels, so the rest of the pipeline can use the standard
    level-based API without change

    All coordinates exposed through this class are normalised
    so that (0, 0) is the top-left corner of the tissue area, matching convention for general class

    Channel order: Zeiss brightfield scanners store pixels as Bgr24 (BGR byte
    order).  This class detects the PixelType field in CZI metadata and
    converts BGR → RGB automatically so callers always receive standard RGB.
    For fluorescence CZIs, the three channels are returned as-is

    Install dependency: pip install aicspylibczi
    """

    _DOWNSAMPLE_TABLE: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

    def __init__(self, path: str | Path) -> None:
        try:
            from aicspylibczi import CziFile
        except ImportError as exc:
            raise ImportError(
                "aicspylibczi is required to read CZI files. "
                "Install it with:  pip install aicspylibczi"
            ) from exc

        self._path = Path(path)
        self._czi = CziFile(str(path))

        # Mosaic bounding box — may have a non-zero origin in CZI space
        bbox = self._czi.get_mosaic_bounding_box()
        self._origin_x: int = bbox.x
        self._origin_y: int = bbox.y
        self._full_w: int = bbox.w
        self._full_h: int = bbox.h

        # Build the simulated pyramid table
        self._downsamples: tuple[float, ...] = tuple(
            float(d) for d in self._DOWNSAMPLE_TABLE
            if self._full_w // d >= 1 and self._full_h // d >= 1
        )
        self._level_dims: tuple[tuple[int, int], ...] = tuple(
            (max(1, self._full_w // int(d)), max(1, self._full_h // int(d)))
            for d in self._downsamples
        )

        # Determine which C index to use (required by read_mosaic for mosaic files)
        dims = self._czi.get_dims_shape()
        c_info = dims[0].get("C") if dims else None
        self._c_start: int = c_info[0] if c_info else 0

        self._is_bgr: bool = self._detect_bgr()

    # Internal helpers 

    def _detect_bgr(self) -> bool:
        """True if CZI PixelType is BGR (needs channel swap before returning RGB).

        czi.meta is already an xml.etree.ElementTree.Element, not a string
        """
        try:
            node = self._czi.meta.find(".//PixelType")
            return node is not None and "Bgr" in (node.text or "")
        except Exception:
            return False

    def _to_rgb_uint8(self, img: np.ndarray) -> np.ndarray:
        """Normalise a raw aicspylibczi array to (H, W, 3) uint8 RGB.

        Handles common output shapes from read_mosaic:
          (1, H, W, C) — single scene, channels last  [most common]
          (H, W, C) — after squeeze
          (C, H, W) — channels first (some versions / configs)
          (H, W) - grayscale
        """
        img = np.squeeze(img)  # drop any length-1 dims (scene, Z, T, …)

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3:
            # Detect channels-first layout: shape[0] is small, other dims are large
            if img.shape[-1] not in (1, 3, 4) and img.shape[0] in (1, 3, 4):
                img = np.moveaxis(img, 0, -1)
            # Normalise channel count to 3
            if img.shape[-1] == 1:
                img = np.concatenate([img, img, img], axis=-1)
            elif img.shape[-1] >= 4:
                img = img[:, :, :3]
        else:
            raise ValueError(
                f"Unexpected array shape from aicspylibczi: {img.shape}. "
                "Only 2-D or 3-D images are supported."
            )

        # Normalise dtype to uint8
        if img.dtype == np.uint8:
            pass
        elif img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        # BGR → RGB for Zeiss brightfield pixel type
        if self._is_bgr:
            img = img[:, :, ::-1].copy()

        return self._fill_unscanned(img)

    # SlideReader interface 

    @property
    def dimensions(self) -> tuple[int, int]:
        return (self._full_w, self._full_h)

    @property
    def level_count(self) -> int:
        return len(self._downsamples)

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return self._level_dims

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return self._downsamples

    def get_mpp(self) -> float:
        """Parse microns-per-pixel from CZI XML metadata.

        Zeiss stores the X pixel size in metres under:
          Scaling/Items/Distance[@Id="X"]/Value

        czi.meta is already an xml.etree.ElementTree.Element — do not re-parse.
        Value is in metres; multiply by 1e6 to get microns.
        """
        try:
            root = self._czi.meta  # already an Element, not a string
            for xpath in (
                './/Scaling/Items/Distance[@Id="X"]/Value',
                './/Metadata/Scaling/Items/Distance[@Id="X"]/Value',
            ):
                node = root.find(xpath)
                if node is not None and node.text and node.text.strip():
                    return float(node.text) * 1e6  # metres → microns
        except Exception:
            pass
        raise ValueError(
            f"Cannot determine MPP from CZI metadata in '{self._path.name}'. "
            "Check scanner calibration settings."
        )

   
    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Largest level whose downsample factor does not exceed *downsample*."""
        best = 0
        for i, d in enumerate(self._downsamples):
            if d <= downsample:
                best = i
        return best

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        """Read a region and return (H, W, 3) uint8 RGB.

        Args:
            location: (x, y) top-left in level-0 pixels, 0-based from the
                      normalised tissue origin (bbox offset handled internally).
            level:    Simulated pyramid level (0 = full resolution).
            size:     (width, height) at the requested level.
        """
        x0, y0 = location
        w, h = size
        downsample = self._downsamples[level]

        # Translate to CZI mosaic coordinate space (add bbox origin)
        czi_x = x0 + self._origin_x
        czi_y = y0 + self._origin_y
        # Region extents at full resolution (level 0)
        czi_w = max(1, round(w * downsample))
        czi_h = max(1, round(h * downsample))

        img = self._czi.read_mosaic(
            region=(czi_x, czi_y, czi_w, czi_h),
            scale_factor=1.0 / downsample,
            C=self._c_start,
        )
        return self._to_rgb_uint8(img)

    def get_thumbnail(self, max_size: int = 2000) -> np.ndarray:
        scale = max_size / max(self._full_w, self._full_h)
        img = self._czi.read_mosaic(scale_factor=scale, C=self._c_start)
        return self._to_rgb_uint8(img)

    def close(self) -> None:
        pass  # aicspylibczi does not require explicit close



# Manifest


def create_manifest(
    slide_dir: str | Path,
    out_path: str | Path = "manifest.csv",
    thickness_um: list[float] | None = None,
) -> Path:
    """Write a manifest CSV from a folder of slide files in filesystem order.

    Row order = section order. Verify the CSV before running the pipeline.

    Args:
        slide_dir: folder containing slide files.
        out_path:  where to write the manifest (default: manifest.csv in cwd).
        thickness_um: optional per-section thickness values (microns), one
            per file in the same natural-sorted order the rows are written
            in. When provided, a 4th 'thickness_um' column is appended to
            the manifest. When None
            (default), the manifest is written with the original 3-column
            header — byte-identical to the pre-existing format.

    Returns:
        Path to the written CSV.

    Raises:
        FileNotFoundError: slide_dir does not exist
        ValueError: no supported slide files found, or thickness_um is
            provided with a length that does not match the number of files.
    """
    slide_dir = Path(slide_dir)
    if not slide_dir.exists():
        raise FileNotFoundError(f"Slide directory not found: {slide_dir}")

    supported = {".czi", ".svs", ".ndpi", ".tif", ".tiff", ".scn", ".mrxs"}
    files = [p for p in slide_dir.iterdir() if p.suffix.lower() in supported]
    if not files:
        raise ValueError(f"No supported slide files found in {slide_dir}")

    # Natural sort: split filenames into text/number chunks so that
    # e.g. section_10 sorts after section_9, not after section_1.
    def _natural_key(p: Path) -> list:
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", p.name)]
    files.sort(key=_natural_key)

    if thickness_um is not None and len(thickness_um) != len(files):
        raise ValueError(
            f"thickness_um has {len(thickness_um)} values but {len(files)} "
            f"slide files were found in {slide_dir}."
        )

    out_path = Path(out_path)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        if thickness_um is None:
            writer.writerow(["section_index", "filename", "path"])
            for i, p in enumerate(files):
                writer.writerow([i, p.name, str(p)])
        else:
            writer.writerow(["section_index", "filename", "path", "thickness_um"])
            for i, (p, thickness) in enumerate(zip(files, thickness_um)):
                writer.writerow([i, p.name, str(p), thickness])

    return out_path


def load_manifest(csv_path: str | Path) -> list[Path]:
    """Load slide paths in section order from a pre-ordered manifest CSV.
    Manifest must have "path" column

    Returns:
        List of Path objects in section order (row 0 = section 0).

    Raises:
        FileNotFoundError: manifest file does not exist
        KeyError: manifest has no 'path' column
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "path" not in (reader.fieldnames or []):
            raise KeyError("Manifest CSV must have a 'path' column.")
        return [Path(row["path"]) for row in reader]


def load_manifest_thickness(csv_path: str | Path) -> list[float | None]:
    """Load per-section thickness (microns) from a manifest CSV, in row order.

    The 'thickness_um' column is optional : when present it must be a positive
    float per row; when absent, one None is returned per row instead of
    raising. Row order matches load_manifest exactly — both iterate the
    same csv.DictReader with no reordering, so zipping their results yields
    aligned (path, thickness) pairs.

    Returns:
        List of floats (or None where the cell is blank) in section order,
        or a list of None (one per row) when the column is absent entirely.

    Raises:
        FileNotFoundError: manifest file does not exist.
        ValueError: thickness_um <= 0.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if "thickness_um" not in (reader.fieldnames or []):
            return [None for _ in reader]

        thicknesses: list[float | None] = []
        for row in reader:
            raw = (row.get("thickness_um") or "").strip()
            if not raw:
                thicknesses.append(None)
                continue
            value = float(raw)
            if value <= 0:
                raise ValueError(f"thickness_um must be positive, got {value}")
            thicknesses.append(value)
        return thicknesses


# Factory


def open_slide(path: str | Path) -> SlideReader:
    """Open a whole-slide image and return the appropriate SlideReader.

    Routing:
        .czi  → CziReader  
        other → OpenSlideReader

    Raises:
        FileNotFoundError: path does not exist.
        ImportError: CZI file given but aicspylibczi is not installed.
        openslide.OpenSlideUnsupportedFormatError: format not recognised by OpenSlide.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Slide file not found: {p}")
    if p.suffix.lower() == ".czi":
        return CziReader(p)
    else:
        return OpenSlideReader(p)



# Module-level convenience wrappers (keep old call sites working)


def get_mpp(slide: SlideReader) -> float:
    """Microns per pixel at level 0."""
    return slide.get_mpp()


def best_level_for_mpp(slide: SlideReader, target_mpp: float) -> int:
    """Pyramid level whose resolution is closest to *target_mpp* µm/px."""
    return slide.best_level_for_mpp(target_mpp)


def read_region_rgb(
    slide: SlideReader,
    location: tuple[int, int],
    level: int,
    size: tuple[int, int],
) -> np.ndarray:
    """Read a region and return (H, W, 3) uint8 RGB."""
    return slide.read_region(location, level, size) # ensure this is RGB and not RGBA. use .convert("RGB")


def iter_tiles(
    slide: SlideReader,
    level: int,
    tile_size: int = 1024,
    overlap: int = 0,
) -> Generator[tuple[tuple[int, int], tuple[int, int], np.ndarray], None, None]:
    """Yield tiles covering the full slide. See SlideReader.iter_tiles."""
    yield from slide.iter_tiles(level, tile_size=tile_size, overlap=overlap)


def get_thumbnail(slide: SlideReader, max_size: int = 2000) -> np.ndarray:
    """Return a downsampled (H, W, 3) uint8 RGB thumbnail."""
    return slide.get_thumbnail(max_size=max_size)
