#!usr/bin/env python3
# -*- coding: utf-8 -*-

""" File containing tests for functions that I have written """
from math import atan2

import numpy as np
import pytest
from scipy.ndimage import center_of_mass
from skimage.draw import ellipsoid
from skimage.transform import rotate

from nm_alignment_by_pca import find_major_axis_by_pca
from nm_alignment_basic import calculate_alignment_angle_2d
from utils import rotate_image_2d


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
