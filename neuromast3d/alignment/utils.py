#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from aicsimageio import AICSImage
import numpy as np
from skimage.transform import rotate
from sklearn.decomposition import PCA


def rotate_image_2d(image: np.array, angle: float, interpolation_order: int = 0):
    """ Source: aicsshparam/shtools.py """

    if image.ndim != 4:
        raise ValueError(f'Invalid shape {image.shape} of input image.')

    if not isinstance(interpolation_order, int):
        raise ValueError('Only integer values are accepted for interpolation order.')

    # Make z to be the last axis. Required for skimage rotation.
    image = np.swapaxes(image, 1, 3)

    img_aligned = []
    for stack in image:
        stack_aligned = rotate(
                image=stack,
                angle=-angle,
                resize=True,
                order=interpolation_order,
                preserve_range=True
        )
        img_aligned.append(stack_aligned)
    img_aligned = np.array(img_aligned)

    img_aligned = np.swapaxes(img_aligned, 1, 3)
    img_aligned = img_aligned.astype(image.dtype)

    return img_aligned


def unit_vector(vector):
    uvec = vector / np.linalg.norm(vector)
    return uvec


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.archos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return angle


def get_membrane_segmentation(path_to_seg):
    seg_mem = AICSImage(path_to_seg).data.squeeze()
    return seg_mem


def prepare_vector_for_napari(vector, origin, scale=1):
    if len(vector.shape) < 2:
        vector = np.expand_dims(vector, axis=0)
    elif len(vector.shape) == 2:
        vector = vector
    else:
        raise ValueError(f'Expected ndims == 1 or 2, \
                        but vector has ndims == {len(vector.shape)}')
    origin = np.array(origin)
    origin = np.reshape(origin, vector.shape)
    vector_start = np.tile(origin, (1,))
    vector_end = vector*scale
    napari_vector = np.stack((vector_start, vector_end), axis=1)
    return napari_vector


def find_major_axis_by_pca(image, threed=False):
    if threed:

        # Find three major axes
        pca = PCA(n_components=3)

        # image = image.reshape(1, *image.shape)

        # Find coordinates where image has a nonzero value (i.e. is not bg)
        # Assumes image has been read in by ZYX dimension order
        z, y, x = np.nonzero(image)

        # The 'xyz' object has final shape (N, 3) where N is the number
        # of coordinates where the image has a nonzero value and the three
        # columns coorrespond to x, y, and z coordinates respectively
        xyz = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])

        # Major axes returned as ndarray of shape (3, 3)
        # Each row vector = a major axis, arranged in descending order
        # Each column corresponds to x, y, and z values respectively
        pca = pca.fit(xyz)
        eigenvecs = pca.components_

    else:
        pca = PCA(n_components=2)
        # image = image.reshape(1, *image.shape)
        print(image.shape)
        z, y, x = np.nonzero(image)
        xy = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
        pca = pca.fit(xy)
        eigenvecs = pca.components_
    return eigenvecs

