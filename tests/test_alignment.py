#!usr/bin/env python3
# -*- coding: utf-8 -*-

from math import atan2

import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.draw import ellipsoid
from skimage.transform import rotate

from neuromast3d.alignment.utils import find_major_axis_by_pca, rotate_image_2d_custom
from neuromast3d.alignment.nm_alignment_basic import align_cell_xz_long_axis_to_z_axis, calculate_2d_long_axis_angle_to_z_axis, calculate_alignment_angle_2d, calculate_alignment_angles, normalize_centroid
from neuromast3d.prep_single_cells.prep_single_cells import create_cropping_roi, crop_to_roi


@pytest.fixture
def orig_ellipsoid():
    return ellipsoid(51, 101, 201)


@pytest.fixture
def rotated_ellipsoid(orig_ellipsoid):
    # assume ZYX order
    # rotate around x, y, and z axes
    ellip = orig_ellipsoid
    ellip_rot1 = ndi.rotate(ellip, 30, (0, 1), order=0)
    ellip_rot2 = ndi.rotate(ellip_rot1, 45, (0, 2), order=0)
    ellip_rot3 = ndi.rotate(ellip_rot2, 60, (1, 2), order=0)
    return ellip_rot1, ellip_rot2, ellip_rot3


def test_find_major_axis_by_pca(orig_ellipsoid):
    """ Test that finding major axis by PCA works on an ellipsoid

    Note that the function assumes input array dims are ordered as ZYX,
    opposite the draw.ellipsoid convention.

    Also, the order of coords is flipped during the function, such that
    the columns of the resulting eigenvector matrix represent x, y, z.
    """

    ellip = orig_ellipsoid
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
    centroid_normed = normalize_centroid(image, origin)
    angle = calculate_alignment_angle_2d(
            centroid_normed=centroid_normed
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


def test_align_cell_xz_long_axis_to_z_axis_spheres(concentric_spheres):
    # rotating spheres should not change anything
    aligned, _ = align_cell_xz_long_axis_to_z_axis(concentric_spheres, concentric_spheres)
    np.testing.assert_allclose(aligned, concentric_spheres)


def test_align_cell_xz_long_axis_to_z_axis_rot_ellip(rotated_ellipsoid):
    ellip_rot1, ellip_rot2, _ = rotated_ellipsoid
    # To account differences in the image shape post rotation,
    # we rotate twice for 45 degrees rather than once for 90
    expected_pre = ndi.rotate(ellip_rot1, 45, (0, 2), order=0)
    expected = ndi.rotate(expected_pre, 45, (0, 2), order=0)[np.newaxis, :, :, :]
    ellip_rot1 = ellip_rot1[np.newaxis, :, :, :]
    ellip_rot2 = ellip_rot2[np.newaxis, :, :, :]
    aligned, _ = align_cell_xz_long_axis_to_z_axis(ellip_rot2, ellip_rot2)
    np.testing.assert_allclose(aligned, expected)


@pytest.mark.parametrize(
    'init_angle, expected_angle, axes, proj_type', [
        (0, -90, (0,2), 'xz'),
        (15, 75, (0,2), 'xz'),
        (45, 45, (0,2), 'xz'),
        (90, 0, (0,2), 'xz'),
        (-15, -75, (0,2), 'xz'),
        (15, 75, (0,1), 'yz')
    ]
)
def test_calculate_2d_long_axis_angle_to_z_axis(orig_ellipsoid, init_angle, expected_angle, axes, proj_type):
    ellip_rotated = ndi.rotate(orig_ellipsoid, init_angle, axes, order=0)
    ellip_rotated = ellip_rotated[np.newaxis, :, :, :]
    actual_angle = calculate_2d_long_axis_angle_to_z_axis(ellip_rotated, proj_type)
    np.testing.assert_almost_equal(actual_angle, expected_angle, decimal=2)


@pytest.mark.parametrize(
    'mode, origin, expected', [
        ('unaligned', (0, 0, 0), (0, 0, 0)),
        ('principal_axes', (0, 0, 0), (-60, -45, -30)),
        ('xy_only', (0, 0, 0), (-45, 0, 0))
    ]
)
def test_calculate_alignment_angles(rotated_ellipsoid, mode, origin, expected):
    _, _, ellip_rot3 = rotated_ellipsoid
    centroid_normed = normalize_centroid(ellip_rot3, origin)
    ellip_rot3 = ellip_rot3[np.newaxis, :, :, :]
    angles = calculate_alignment_angles(img=ellip_rot3, mode=mode, use_channels=0, centroid_normed=centroid_normed)
    np.testing.assert_allclose(angles, expected, rtol=1e-02, atol=5)
