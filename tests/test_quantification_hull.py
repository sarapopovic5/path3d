"""Flat (one-section-thick) components: no convex hull warning, and solidity
from their corner hull rather than a blanket 1.0."""

import warnings

import numpy as np

import path3d.quantification as qt


def _flat_l_shape():
    mask = np.zeros((1, 4, 4), bool)  # one z-plane: Qhull cannot hull it in 3D
    mask[0, :, 0] = True
    mask[0, 3, :] = True
    return mask


def test_flat_component_has_no_hull_image_and_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        assert qt._component_hull_image(_flat_l_shape()) is None


def test_flat_l_shape_solidity_is_below_one():
    mask = _flat_l_shape()
    voxel_um = 8.0
    volume = mask.sum() * voxel_um**3
    d = qt._component_shape_descriptors(mask, voxel_um, volume)
    # 7 voxels of a 4x4 square's corner hull: well short of convex
    assert 0.0 < d["solidity"] < 1.0
    assert d["feret_diameter_max_um"] > 0


def test_flat_rectangle_is_still_fully_solid():
    mask = np.ones((1, 3, 5), bool)
    voxel_um = 8.0
    d = qt._component_shape_descriptors(mask, voxel_um, mask.sum() * voxel_um**3)
    assert np.isclose(d["solidity"], 1.0)


def test_3d_component_hull_is_unchanged():
    mask = np.zeros((4, 6, 6), bool)
    mask[1:3, 1:5, 1:5] = True
    hull = qt._component_hull_image(mask)
    assert hull is not None and (hull == mask).all()
