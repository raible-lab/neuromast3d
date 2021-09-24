#!usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage as ndi

from neuromast3d.alignment.utils import find_major_axis_by_pca


def rotate_image_3d(mask):
    """Calculate 3D rotation angles

    Note: Currently, this rotation acts to align the three major axes of the
    image (as calculated by PCA) to be the new x, y, and z coordinates.

    Parameters
    ----------
    mask : np.array
        The mask to use to calculate major axes and rotation angles.
        Must have ndims = 3.

    """

    if mask.ndim != 3:
        raise ValueError('Image must have ndims == 3. '
                         f'Provided image has ndims of {mask.ndim}')

    # Calculate principal axes of the image
    eigenvecs = find_major_axis_by_pca(mask, threed=True)

    # Rotate around z axis first (yaw angle)
    angle_1 = np.arctan(eigenvecs[0][1] / eigenvecs[0][0]) * 180 / np.pi
    img_rot_1 = ndi.rotate(mask, angle_1, (1, 2), reshape=True, order=0)

    # Rotate around y axis next (pitch angle)
    eigenvecs_2 = find_major_axis_by_pca(img_rot_1, threed=True)
    angle_2 = np.arctan(eigenvecs_2[0][2] / eigenvecs_2[0][0]) * 180 / np.pi
    img_rot_2 = ndi.rotate(img_rot_1, angle_2, (0, 2), reshape=True, order=0)

    # Rotate around x axis last (roll angle)
    eigenvecs_3 = find_major_axis_by_pca(img_rot_2, threed=True)
    angle_3 = np.arctan(eigenvecs_3[1][2] / eigenvecs_3[1][1]) * 180 / np.pi

    return (angle_1, angle_2, angle_3)


def apply_3d_rotation(image, yaw, pitch, roll):

    """Apply a 3D rotation to an image. Implemented as a sequence of three
    rotations using Euler-like angles (I am not sure if I am doing it
    completely according to some standard, but it does seem to work.)

    Parameters
    ----------
    image : np.array
        The image to be rotated. Must have ndims == 4.
        The dimension order expected is CZYX.

    yaw : float
        The angle in degrees to rotate around the z axis during the first
        rotation.

    pitch : float
        The angle in degrees to rotate around the y axis during the second
        rotation.

    roll : float
        The angle in degrees to rotate around the x axis during the third
        rotation.

    """

    if image.ndim != 4:
        raise ValueError('Image must have ndims == 4.'
                         f'Provided image has ndims of {image.ndim}')

    # Rotate each channel independently, then combine at the end
    img_aligned = []
    for channel in image:
        img_rot = ndi.rotate(channel, yaw, (1, 2), reshape=True, order=0)
        img_rot = ndi.rotate(img_rot, pitch, (0, 2), reshape=True, order=0)
        img_rot = ndi.rotate(img_rot, roll, (0, 1), reshape=True, order=0)
        img_aligned.append(img_rot)
    img_aligned = np.array(img_aligned)

    # Make same as input image dtype
    img_aligned = img_aligned.astype(image.dtype)

    return img_aligned
