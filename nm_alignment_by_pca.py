#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from aicsimageio import AICSImage
from aicsimageprocessing import resize, resize_to
import napari
import numpy as np
from skimage.morphology import binary_closing, ball
from scipy.ndimage import center_of_mass
from sklearn.decomposition import PCA

# Constants (note: these are guessed right now)
PixelScaleZ = 0.224
PixelScaleX = 0.0497
PixelScaleY = PixelScaleX
standard_res_qcb = PixelScaleX


def get_largest_cc(image):
    largest_cc = np.argmax(np.bincount(image.flat))
    return largest_cc


def switch_label_values(label_image, first, second):
    last_label = label_image.max()
    label_image = np.where(label_image == first, last_label + 1, label_image)
    label_image = np.where(label_image == second, first, label_image)
    label_image = np.where(label_image == last_label + 1, second, label_image)
    return label_image


project_dir = '/home/maddy/projects/claudin_gfp_5dpf_airy_live/'
img_id = '20200617_1-O1'

# Read in raw and segmented images
raw_reader = AICSImage(f'{project_dir}/stack_aligned/{img_id}.tiff')
raw_img = raw_reader.get_image_data('ZYX', S=0, T=0, C=0)

seg_reader = AICSImage(f'{project_dir}/label_images/{img_id}_rawlabels.tiff')
seg_img = seg_reader.get_image_data('ZYX', S=0, T=0, C=0)

# Interpolate along z to create isotropic voxel dimensions
# (same as preparing single cells for cvapipe_analysis)

raw_img = resize(
        raw_img,
        (
            PixelScaleZ / standard_res_qcb,
            PixelScaleY / standard_res_qcb,
            PixelScaleX / standard_res_qcb
        ),
        method='bilinear'
    ).astype(np.uint16)

seg_img = resize(
        seg_img,
        (
            PixelScaleZ / standard_res_qcb,
            PixelScaleY / standard_res_qcb,
            PixelScaleX / standard_res_qcb
        ),
        method='bilinear'
    ).astype(np.uint16)

# Fix segmentation so background label is 0
# This assumes the largest connected component is the background
lcc = get_largest_cc(seg_img)
seg_img = switch_label_values(seg_img, 0, lcc)

# Merge cell labels together to create whole neuromast (nm) mask
nm = seg_img > 0
nm = nm.astype(np.uint8)
nm = nm*255

# Clean up the neuromast mask (could investigate other functions here)
nm = binary_closing(nm, ball(5))

# Use PCA to find major axis of 3D binary mask
pca = PCA(n_components=3)
nm_vals = nm.reshape(1, *nm.shape)
z, y, x = np.nonzero(nm)
xyz = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])
pca = pca.fit(xyz)
eigenvecs = pca.components_

# Scale up the eigenvecs for visualization
vector_endpoints = np.array(eigenvecs)*100
nm_centroid = np.array(center_of_mass(nm))

num_dims = eigenvecs.shape[1]
vector_startpoints = np.tile(nm_centroid, (3, 1))

assert vector_startpoints.shape == vector_endpoints.shape

major_axes = np.stack((vector_startpoints, vector_endpoints), axis=1)
print(major_axes)