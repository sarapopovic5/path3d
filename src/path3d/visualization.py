"""Interactive 3D viewing and headless screenshot export for path3d volumes.

Renders the SpatialData container from ``path3d.volume.build_volume``
(12 um isotropic "tissue_labels" + optional "nuclei" table, "microns_3d"
coord system). ``view_volume()`` opens an interactive napari-spatialdata
session; ``render_volume_screenshot()`` writes a single PNG headlessly.
Both share layer styling via ``_build_layers``/``_add_legend``.

``_TISSUE_3D_COLORS`` is this module's own 3D palette, deliberately separate
from the 2D QC-overlay colors in ``path3d.config`` — do not import that
constant here.

napari/napari_spatialdata/qtpy/spatialdata are optional (``pip install
path3d[viz]``) and imported lazily so importing this module needs no GUI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from path3d.config import TISSUE_CLASSES

# Deliberate, napari-3D-specific palette. Class 0 ("space") is fully transparent so a solid
# background does not dominate the render
_TISSUE_3D_COLORS: dict[int, np.ndarray] = {
    0: np.array([0.0, 0.0, 0.0, 0.0]),  # space -- fully transparent
    1: np.array([0.90, 0.10, 0.10, 1.0]),  # Epithelium
    2: np.array([0.10, 0.45, 0.90, 1.0]),  # Stroma
    3: np.array([0.20, 0.75, 0.30, 1.0]),  # rbc
    4: np.array([0.95, 0.55, 0.10, 1.0]),  # Intraluminal secretion
}

_DEFAULT_POINT_COLORMAP = "viridis"
_DEFAULT_COLOR_BY = "area_um2"
_LEGEND_WIDGET_NAME = "Tissue classes"


def view_volume(
    sdata_or_path: "Any | str | Path",
    *,
    headless: bool = False,
    tissue_type: str = "HGSC",
    color_by: str = _DEFAULT_COLOR_BY,
    point_size: float = 15.0,
) -> "Any":
    """Open an assembled 3D volume in an interactive napari-spatialdata session.

    Populates the canvas via ``_build_layers``/``_add_legend`` (not
    ``Interactive``'s own loader, which would use napari's default label
    colormap instead of ``_TISSUE_3D_COLORS``), and forces 3D display since
    napari defaults to a flat 2D slice.

    Args:
        sdata_or_path: Path to a SpatialData zarr store, or an already-loaded
            SpatialData object.
        headless: Reserved for future use; currently has no effect. Use
            ``render_volume_screenshot`` for a real headless path.
        tissue_type: Key into ``path3d.config.TISSUE_CLASSES``.
        color_by: ``.obs`` column on the nuclei table to color points by.
        point_size: Point size passed to ``add_points``.

    Returns:
        The ``napari_spatialdata.Interactive`` instance 

    Raises:
        ImportError: napari-spatialdata (or spatialdata) is not installed.
    """
    sdata = _load_sdata(sdata_or_path)

    try:
        from napari_spatialdata import Interactive
    except ImportError as exc:
        raise ImportError(
            "napari-spatialdata is required. Install: pip install path3d[viz]"
        ) from exc

    # headless=True suppresses Interactive.__init__'s own blocking napari.run(); the caller runs it instead.
    interactive = Interactive(sdata, headless=True)

    # `Interactive` exposes no `ndisplay` constructor argument and never
    # loads any element into the canvas on its own- resolve the underlying napari Viewer ourselves so we
    # can force 3D display and populate it via the shared layer-construction
    # helpers.
    viewer = getattr(interactive, "_viewer", None)
    if viewer is None:
        try:
            import napari
        except ImportError as exc:
            raise ImportError(
                "napari is required. Install: pip install path3d[viz]"
            ) from exc
        viewer = napari.current_viewer()

    if viewer is not None:
        viewer.dims.ndisplay = 3
        _build_layers(
            viewer,
            sdata,
            tissue_type=tissue_type,
            color_by=color_by,
            point_size=point_size,
        )
        _add_legend(viewer, tissue_type=tissue_type)
    # else: no viewer could be resolved (e.g. a future napari-spatialdata
    # release renames the private `_viewer` attribute) -- degrade to browsable-panel behavior 

    return interactive


def _load_sdata(sdata_or_path: "Any | str | Path") -> "Any":
    """Load a SpatialData object from a zarr path, or pass an object through.

    Raises:
        FileNotFoundError: ``sdata_or_path`` is a path that does not exist.
        ImportError: ``spatialdata`` is not installed and a path was given.
    """
    if isinstance(sdata_or_path, (str, Path)):
        path = Path(sdata_or_path)
        if not path.exists():
            raise FileNotFoundError(f"SpatialData zarr not found: {path}")
        try:
            import spatialdata
        except ImportError as exc:
            raise ImportError(
                "spatialdata is required. Install: pip install path3d[viz]"
            ) from exc
        return spatialdata.read_zarr(str(path))
    return sdata_or_path


def _voxel_scale_from_affine(
    label_element: "Any", coord_system: str = "microns_3d"
) -> tuple[float, float, float]:
    """Read the ``(z, y, x)`` voxel size in microns from the stored affine.

    Voxel size comes from ``build_volume``'s ``target_voxel_um`` and must be
    read from the affine

    Raises:
        ImportError: spatialdata is not installed
    """
    try:
        from spatialdata.transformations import get_transformation
    except ImportError as exc:
        raise ImportError(
            "spatialdata is required. Install: pip install path3d[viz]"
        ) from exc

    affine = get_transformation(label_element, coord_system)
    if isinstance(affine, dict):
        # get_transformation's return type is a Union keyed on get_all; with
        # get_all=False (the default) and an explicit coord_system it always
        # returns a single BaseTransformation, never this dict branch.
        raise TypeError(
            f"get_transformation returned a dict of transformations for "
            f"coordinate system {coord_system!r}; expected a single "
            f"transformation (call with get_all=False)."
        )
    matrix = affine.to_affine_matrix(
        input_axes=("z", "y", "x"), output_axes=("z", "y", "x")
    )
    return (float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[2, 2]))


def _build_layers(
    viewer: "Any",
    sdata: "Any",
    *,
    tissue_type: str = "HGSC",
    color_by: str = _DEFAULT_COLOR_BY,
    point_size: float = 15.0,
) -> list[str]:
    """Add the tissue Labels layer, one hidden per-class Labels layer, and the
    nuclei Points layer.

    Shared by ``view_volume()`` and ``render_volume_screenshot()`` so layer
    styling is defined exactly once.

    Returns:
        Names of the layers added: the combined ``"tissue_labels"`` layer,
        then one ``"class_<name>"`` layer per non-background tissue class
        (hidden by default so the default render is unchanged), then
        ``"nuclei"`` if a nuclei table is present, e.g.
        ``["tissue_labels", "class_Epithelium", ..., "nuclei"]``.

    Raises:
        KeyError: No "tissue_labels" element in ``sdata.labels``.
        ImportError: ``napari`` or ``spatialdata`` is not installed.
    """
    if "tissue_labels" not in sdata.labels:
        raise KeyError(
            "No 'tissue_labels' element in SpatialData container "
            f"(labels: {sorted(sdata.labels)})"
        )

    label_element = sdata.labels["tissue_labels"]
    scale = _voxel_scale_from_affine(label_element)

    try:
        from napari.utils.colormaps import direct_colormap
    except ImportError as exc:
        raise ImportError(
            "napari is required. Install: pip install path3d[viz]"
        ) from exc

    # label_element.data is a dask array; pass it straight through so napari
    # consumes it lazily -- do not .compute()/np.asarray() it here.
    viewer.add_labels(
        label_element.data,
        name="tissue_labels",
        colormap=direct_colormap(_TISSUE_3D_COLORS),
        scale=scale,
    )
    layer_names = ["tissue_labels"]

    # One hidden-by-default boolean-mask layer per non-background class, so
    # the layer list gains checkbox-toggleable single-class views without
    # changing the default (combined) render. Prefixed "class_" (not the
    # bare class name) because napari_spatialdata's SdataWidget renames any
    # layer whose name collides with a SpatialData element name (appending
    # "_external"), and bare class names could collide with future elements.
    for class_index, class_name in sorted(TISSUE_CLASSES[tissue_type].items()):
        if class_index == 0:
            continue
        # Lazy dask comparison + astype -- stays lazy, never .compute()/np.asarray().
        mask = (label_element.data == class_index).astype(np.uint8)
        class_colormap = direct_colormap(
            {0: _TISSUE_3D_COLORS[0], 1: _TISSUE_3D_COLORS[class_index]}
        )
        layer_name = f"class_{class_name}"
        viewer.add_labels(
            mask,
            name=layer_name,
            colormap=class_colormap,
            scale=scale,
            visible=False,
        )
        layer_names.append(layer_name)

    if "nuclei" in sdata.tables:
        adata = sdata.tables["nuclei"]
        coords = np.asarray(adata.obsm["spatial_3d"])
        values = np.asarray(adata.obs[color_by], dtype=float)
        viewer.add_points(
            coords,
            name="nuclei",
            properties={color_by: values},
            face_color=color_by,
            face_colormap=_DEFAULT_POINT_COLORMAP,
            size=point_size,
            out_of_slice_display=True,
        )
        layer_names.append("nuclei")

    return layer_names


def _add_legend(viewer: "Any", *, tissue_type: str = "HGSC") -> "Any":
    """Dock an in-viewer legend mapping each tissue class to its color.

    Uses a Qt dock widget rather than canvas overlay so it's captured by
    ``viewer.screenshot(canvas_only=False)`` too.

    Raises:
        ImportError: ``qtpy`` (with a Qt backend) is not installed.
    """
    try:
        from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
    except ImportError as exc:
        raise ImportError(
            "qtpy (with a Qt backend) is required. Install: pip install path3d[viz]"
        ) from exc

    widget = QWidget()
    layout = QVBoxLayout()
    for class_index, class_name in sorted(TISSUE_CLASSES[tissue_type].items()):
        label = QLabel(f"{class_index}: {class_name}")
        r, g, b, a = _TISSUE_3D_COLORS[class_index]
        label.setStyleSheet(
            f"background-color: rgba({int(r * 255)}, {int(g * 255)}, "
            f"{int(b * 255)}, {int(a * 255)});"
        )
        layout.addWidget(label)
    widget.setLayout(layout)

    viewer.window.add_dock_widget(widget, area="right", name=_LEGEND_WIDGET_NAME)
    return widget


def render_volume_screenshot(
    sdata_or_path: "Any | str | Path",
    output_path: "str | Path",
    *,
    tissue_type: str = "HGSC",
    color_by: str = _DEFAULT_COLOR_BY,
    point_size: float = 15.0,
    offscreen: bool = True,
) -> Path:
    """Render a SpatialData volume and write a single PNG screenshot to disk.

    Builds a plain, non-interactive ``napari.Viewer``, adds the tissue/nuclei
    layers and legend, screenshots, then closes the viewer.

    Args:
        offscreen: When True (default), sets ``QT_QPA_PLATFORM=offscreen``
            first. If this segfaults on macOS arm64, pass ``offscreen=False``
            and run with a real display.

    Returns:
        Path to the written PNG

    Raises:
        TypeError: ``output_path`` is not a str or Path.
        FileNotFoundError: ``sdata_or_path`` is a path that does not exist.
        KeyError: No "tissue_labels" element in the SpatialData container.
        ImportError: ``napari``, ``qtpy``, or ``spatialdata`` is not
            installed.
    """
    if not isinstance(output_path, (str, Path)):
        raise TypeError(
            f"output_path must be str or Path, got {type(output_path).__name__}"
        )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    sdata = _load_sdata(sdata_or_path)

    try:
        import napari
    except ImportError as exc:
        raise ImportError(
            "napari is required. Install: pip install path3d[viz]"
        ) from exc

    viewer = napari.Viewer(show=False, ndisplay=3)
    try:
        _build_layers(
            viewer,
            sdata,
            tissue_type=tissue_type,
            color_by=color_by,
            point_size=point_size,
        )
        _add_legend(viewer, tissue_type=tissue_type)
        viewer.screenshot(path=str(out), canvas_only=False, flash=False)
    finally:
        viewer.close()

    return out
