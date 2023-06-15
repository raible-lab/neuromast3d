#!usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import numpy as np
import pytest

from neuromast3d.prep_single_cells.prep_single_cells import apply_function_to_all_channels, execute_step, inherit_labels, create_cropping_roi


def test_inherit_labels(small_sphere, big_sphere):
    parent = big_sphere * 42
    child = small_sphere * 7
    child_expected = small_sphere * 42
    child_labeled = inherit_labels(child, parent)
    np.testing.assert_allclose(child_labeled, child_expected)


def test_inherit_labels_empty_child(big_sphere):
    parent = big_sphere * 42
    child_zeros = np.zeros_like(big_sphere)
    child_labeled = inherit_labels(child_zeros, parent)
    # If no nonzero pixels within the mask, should return all zeros
    np.testing.assert_equal(child_labeled, child_zeros)


def test_inherit_labels_child_outside_parent(small_sphere, big_sphere):
    # If some of the "child" pixels are outside the parent
    # Should still work, but raise a warning
    parent = small_sphere * 42
    child = big_sphere * 7
    expected = np.where(small_sphere, 42, 0)
    actual = inherit_labels(child, parent)
    np.testing.assert_equal(actual, expected)
    with pytest.warns(UserWarning):
        warnings.warn("Some child pixels are outside the parent", UserWarning)


def test_create_cropping_roi(small_sphere):
    expected = [1,  33, 0, 33, 0, 33]
    actual = create_cropping_roi(small_sphere)
    np.testing.assert_allclose(actual, expected)


def test_create_cropping_roi_non_3d_input(small_sphere):
    small_sphere_4d = small_sphere[np.newaxis, :, :, :]
    with pytest.raises(NotImplementedError):
        create_cropping_roi(small_sphere_4d)


def test_apply_function_to_all_channels(concentric_spheres):
    # Not sure if np.zeros_like is the best function for testing here
    # Might be better if it were something that only works on 3D images
    expected = np.zeros((2, 33, 33, 33))
    actual = apply_function_to_all_channels(concentric_spheres, np.zeros_like)
    
    np.testing.assert_equal(actual, expected)


def test_apply_function_to_all_channels_non_4d_input(small_sphere):
    # Should fail for non-4D input
    with pytest.raises(ValueError):
        apply_function_to_all_channels(small_sphere, np.zeros_like)

