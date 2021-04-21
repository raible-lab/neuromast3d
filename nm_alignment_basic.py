#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Align all cells to same cardinal direction

As suggested by Lorenzo, this script assumes radial symmetry of the neuromast
and does not try to align cells to an organismal axis (e.g. A/P, D/V).
"""

import argparse
import os
import pathlib
import sys

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
from aicsimageprocessing import resize, resize_to
import numpy as np
import pandas as pd
from scipy.stats import skew
from skimage.morphology import binary_closing, ball
from skimage.measure import regionprops
from skimage.transform import rotate
from scipy.ndimage import center_of_mass

from utils import rotate_image_2d


# Command line arguments
parser = argparse.ArgumentParser(description='Basic cell alignment')
parser.add_argument('project_dir', help='project directory for this run')
parser.add_argument('manifest', help='path to cell manifest in csv format')
parser.add_argument('z_res', help='voxel depth', type=float)
parser.add_argument('xy_res', help='pixel size in xy', type=float)
parser.add_argument(
        '-u',
        '--make_unique',
        help='make the rotation angle unique',
        action='store_true'
)

args = parser.parse_args()

project_dir = args.project_dir
path_to_manifest = args.manifest
z_res = args.z_res
xy_res = args.xy_res

# Check that project directory exists
if not os.path.isdir(project_dir):
    print('Project directory does not exist')
    sys.exit()

# Read the manifest to align cells for this run
cell_df = pd.read_csv(path_to_manifest, index_col=0)

# Create dir to save for this step
step_local_path = f'{project_dir}/alignment'
pathlib.Path(step_local_path).mkdir(parents=True, exist_ok=True)

# Add labels column (TODO: consider adding this step upstream)
cell_ids = cell_df['CellId']
cell_df['label'] = cell_ids.str.split('_', expand=True)[2]

# Create fov dataframe
fov_df = cell_df.copy()
fov_df.drop_duplicates(subset=['fov_id'], keep='first', inplace=True)
fov_df.drop(['CellId', 'crop_raw', 'crop_seg', 'roi'], axis=1, inplace=True)

# Calculate neuromast centroid
nm_centroids = []

# Calculate angles for cell alignment
cell_angles = []

for fov in fov_df.itertuples(index=False):
    seg_reader = AICSImage(fov.fov_seg_path)
    seg_img = seg_reader.get_image_data('ZYX', S=0, T=0, C=0)
    nm = seg_img > 0
    nm = resize(
            seg_img,
            (
                z_res / xy_res,
                xy_res / xy_res,
                xy_res / xy_res
            ),
            method='bilinear'
    )
    seg_img = resize_to(seg_img, nm.shape, method='nearest')
    nm = binary_closing(nm, ball(5))
    nm_centroid = center_of_mass(nm)

    # Save nm centroids matched to fov_id
    nm_centroids.append({'fov_id': fov.fov_id, 'nm_centroid': nm_centroid})

    # Subset cell df for this fov
    current_fov_cells = cell_df[cell_df['fov_id'] == fov.fov_id]

    for cell in current_fov_cells.itertuples(index=False):

        # Actually we were never calculating the cell centroid based on
        # the interpolated image... should we be? That would only throw off
        # the z value I think? We'll try it here
        label = int(cell.label)
        cell_img = np.where(seg_img == label, seg_img, 0)
        cell_centroid = center_of_mass(cell_img)
        cell_centroid = np.subtract(cell_centroid, nm_centroid)

        # Calculate alignment angle in xy plane
        cell_img = cell_img.astype(np.uint8)
        cell_img = cell_img * 255
        z, y, x = np.nonzero(cell_img)

        if args.make_unique:

            # Calculate angle with atan2 to preserve orientation
            # I think this SHOULD align to the 3 o' clock position?
            angle = 180 * np.arctan2(cell_centroid[1], cell_centroid[2]) / np.pi

            # Still don't fully understand what's going on in this section
            # TODO: Write tests to ensure rotation is centered etc.
            x_rot = (x - x.mean()) * np.cos(np.pi * angle / 180) + (
                    y - y.mean()) * np.sin(np.pi * angle / 180)
            xsk = skew(x_rot)
            if xsk < 0.0:
                angle += 180
            angle = angle % 360

        else:

            # Calculate smallest angle
            angle = 0.0
            if np.abs(cell_centroid[2]) > 1e-12:  # avoid divide by zero error ig?
                angle = 180 * np.arctan(cell_centroid[1] / cell_centroid[2]) / np.pi

        # Save angle matched to cell_id
        cell_angles.append({'CellId': cell.CellId, 'rotation_angle': angle})

        # Apply alignment to single cell mask
        reader = AICSImage(cell.crop_seg_path)
        seg_cell = reader.get_image_data('ZYX', S=0, T=0, C=0)

        # Rotate function expects multichannel image
        if seg_cell.ndim == 3:
            seg_cell = np.expand_dims(seg_cell, axis=0)
        cell_aligned = rotate_image_2d(
                image=seg_cell,
                angle=angle,
                interpolation_order=0
        )

        # Save aligned single cell mask
        current_cell_dir = f'{step_local_path}/{fov.fov_id}/{label}'
        pathlib.Path(current_cell_dir).mkdir(parents=True, exist_ok=True)
        seg_path = f'{current_cell_dir}/segmentation.ome.tif'
        crop_seg_aligned_path = pathlib.Path(seg_path)
        writer = ome_tiff_writer.OmeTiffWriter(crop_seg_aligned_path)
        writer.save(cell_aligned, dimension_order='CZYX')

# Add to cell_df
fov_centroid_df = pd.DataFrame(nm_centroids)
cell_df = cell_df.merge(fov_centroid_df, on='fov_id')

# Save angles to cell manifest
angle_df = pd.DataFrame(cell_angles)
cell_df = cell_df.merge(angle_df, on='CellId')
cell_df.to_csv(step_local_path / 'manifest.csv')
