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

from .utils import rotate_image_2d, get_membrane_segmentation, unit_vector, \
        angle_between, find_major_axis_by_pca, prepare_vector_for_napari

# Constants (note: these are guessed right now)
PixelScaleZ = 0.2224
PixelScaleX = 0.0497
PixelScaleY = PixelScaleX
standard_res_qcb = PixelScaleX


project_dir = '/home/maddy/projects/claudin_gfp_5dpf_airy_live/'
nm_id = '20200617_1-SO2'

# Read in raw and segmented images
raw_reader = AICSImage(f'{project_dir}/stack_aligned/{nm_id}.tiff')
raw_img = raw_reader.get_image_data('ZYX', S=0, T=0, C=0)

seg_reader = AICSImage(
        f'{project_dir}/label_images_fixed_bg/{nm_id}_rawlabels.tiff'
)
seg_img = seg_reader.get_image_data('ZYX', S=0, T=0, C=0)

# Merge cell labels together to create whole neuromast (nm) mask
nm = seg_img > 0

# Interpolate along z to create isotropic voxel dimensions
# (same as preparing single cells for cvapipe_analysis)
nm = resize(
        seg_img,
        (
            PixelScaleZ / standard_res_qcb,
            PixelScaleY / standard_res_qcb,
            PixelScaleX / standard_res_qcb
        ),
        method='bilinear'
)

# Clean up the neuromast mask (could investigate other functions here)
nm = binary_closing(nm, ball(5))
nm = nm.astype(np.uint8)
nm = nm*255

# Use PCA to find major axis of 3D binary mask
eigenvecs = find_major_axis_by_pca(nm, threed=True)

# Find centroid of neuromast and each cell
nm_centroid = center_of_mass(nm)
single_cell_props = regionprops(seg_img)

# Vizualize the vector
viz_vector = prepare_vector_for_napari(
        # Needs to be flipped - dims in xyz order but napari expects zyx
        np.flip(eigenvecs[0]),
        origin=(nm_centroid),
        scale=100
)

cell_angles = []
for cell, props in enumerate(single_cell_props):

    # Recreate CellId to ensure rotation applied to correct cell later
    cell_num = cell + 1
    cell_id = f'{nm_id}_{cell_num}'

    # Find centroid position relative to nm_centroid
    cell_centroid = single_cell_props[cell]['centroid']

    # Is this right? Should nm and cell centroid be switched?
    cell_centroid = np.subtract(nm_centroid, cell_centroid)

    # New stuff, testing what difference it makes for 'make_unique'
    cell_img = single_cell_props[cell]['image']
    cell_img = cell_img.astype(np.uint8)
    cell_img = cell_img*255
    z, y, x = np.nonzero(cell_img)

    # Eigenvecs need to be used somewhere in here
    angle = 180.0 * np.arctan2(cell_centroid[1], cell_centroid[2]) / np.pi
    x_rot = (x - x.mean()) * np.cos(np.pi * angle / 180) + (
            y - y.mean()) * np.sin(np.pi * angle / 180)
    xsk = skew(x_rot)
    if xsk < 0.0:
        angle += 180
    angle = angle % 360

    # Save cell angles matched with cell_ids
    cell_angles.append({'CellId': cell_id, 'angle': angle})

# Note: the background issue has been corrected for cvapipe_run_2
# We no longer need to correct for bg labels that were sometimes nonzero

cell_df = pd.read_csv(
        f'{project_dir}/cvapipe_run_2/cell_manifest.csv',
        index_col=0
)
angle_df = pd.DataFrame(cell_angles)
df = cell_df.merge(angle_df, on='CellId')

# Temporarily truncate df for testing
df = df.loc[df['fov_id'] == nm_id]

for row in df.itertuples(index=False):
    seg_cell = get_membrane_segmentation(row.crop_seg)
    seg_cell = np.expand_dims(seg_cell, axis=0)
    angle = row.angle
    seg_rot = rotate_image_2d(seg_cell, angle, interpolation_order=0)
    save_path = f'{project_dir}/cvapipe_run_2/rotation_test/{row.CellId}.ome.tif'
    writer = ome_tiff_writer.OmeTiffWriter(save_path, overwrite_file=True)
    writer.save(seg_rot, dimension_order='ZYX')
    # Question: Do we need to crop the cells again after the rotation?
    # Check if rotation adds padding and if so, if this is a problem.
    # We're using their rotation function so it's probably fine
    # however it is.
    # Also unsure if the cells need to be centered on their centroids
    # Maybe we should test what happens if we do that.
