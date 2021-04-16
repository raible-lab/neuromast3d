#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
from aicsimageprocessing import resize, resize_to
import napari
import numpy as np
import pandas as pd
from scipy.stats import skew
from skimage.morphology import binary_closing, ball
from skimage.measure import regionprops
from skimage.transform import rotate
from scipy.ndimage import center_of_mass
from sklearn.decomposition import PCA


def rotate_image_2d(image, angle, interpolation_order=0):
    if image.ndim != 4:
        raise ValueError(f'Invalid shape {image.shape} of input image.')

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


def get_largest_cc(image):
    largest_cc = np.argmax(np.bincount(image.flat))
    return largest_cc


def switch_label_values(label_image, first, second):
    last_label = label_image.max()
    label_image = np.where(label_image == first, last_label + 1, label_image)
    label_image = np.where(label_image == second, first, label_image)
    label_image = np.where(label_image == last_label + 1, second, label_image)
    return label_image


def unit_vector(vector):
    uvec = vector / np.linalg.norm(vector)
    return uvec


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.archos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return angle


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


def get_membrane_segmentation(path_to_seg):
    seg_mem = AICSImage(path_to_seg).data.squeeze()
    return seg_mem
