#!usr/bin/env python3
# -*- coding: utf-8 -*-

""" File containing tests for functions that I have written """
from math import atan2

import numpy as np
import pytest
from scipy.ndimage import center_of_mass
from skimage.draw import ellipsoid
from skimage.transform import rotate

from neuromast3d.alignment.utils import find_major_axis_by_pca, rotate_image_2d_custom
from neuromast3d.alignment.nm_alignment_basic import calculate_alignment_angle_2d


def test_find_major_axis_by_pca():
    """ Test that finding major axis by PCA works on an ellipsoid

    Note that the function assumes input array dims are ordered as ZYX,
    opposite the draw.ellipsoid convention.

    Also, the order of coords is flipped during the function, such that
    the columns of the resulting eigenvector matrix represent x, y, z.
    """

    ellip = ellipsoid(50, 100, 200)
    eigenvecs = find_major_axis_by_pca(ellip, threed=True)
    eigenvecs = np.absolute(eigenvecs)
    expected = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    np.testing.assert_almost_equal(eigenvecs, expected)

def test_rotate():
    image = np.zeros((5, 5))
    image[3, 2] = 1
    angle = 180 * atan2(-1, 0) / np.pi
    rot_img = rotate(image=image, angle=-angle)
    expected = np.zeros((5, 5))
    expected[2, 3] = 1
    np.testing.assert_almost_equal(rot_img, expected)


@pytest.mark.parametrize(
        'origin, expected_angle', [
            ((0, 0, 0), -45),
            ((11, 0, 0), -45),
            ((0, 11, 0), 0),
            ((0, 0, 11), -90),
            ((0, 22, 0), 45),
            ((0, 0, 22), -135),
            ((0, 23, 23), 135),
            ((0, 23, 11), 90)
        ]
)
def test_calculate_alignment_angle_2d(origin, expected_angle):
    image = ellipsoid(10, 10, 10)
    angle, centroid = calculate_alignment_angle_2d(
            image=image,
            origin=origin,
            make_unique=True
    )
    np.testing.assert_almost_equal(angle, expected_angle)


@pytest.mark.parametrize(
        'a, b,', [
            ((0, True), (0, False)),
            ((90, True), (-90, False)),
            ((180, True), (-180, True)),
            ((90, True), (-270, True))
        ]
)
def test_rotate_image_2d_custom_flipped_sign(a, b):

    """ Test that several conditions that should be equal are

    I.e. that the function behaves as expected with sign
    
    Although the function always uses resize=True currently,
    this test is run with angles that should not change the size/shape
    of the input array.
    """

    x = np.zeros((10, 10))
    x[3, 3] = 1
    x = np.expand_dims(x, axis=(0, 1))
    x_rot_a = rotate_image_2d_custom(
            image=x,
            angle=a[0],
            interpolation_order=0,
            flip_angle_sign=a[1],
    )
    x_rot_b = rotate_image_2d_custom(
            image=x,
            angle=b[0],
            interpolation_order=0,
            flip_angle_sign=b[1],
    )
    np.testing.assert_almost_equal(x_rot_a, x_rot_b)
