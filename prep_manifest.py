#!/usr/bin/env python3
# -*- coding:utf-8 -*-

""" Add a few required columns to manifest.csv before cvapipe_analysis

Ideally, this could happen during the single cell prep stage, but since I
didn't do it then for this run, I'll do it after the fact.

The ones I need to add are structure_name and name_dict. These should be the
same for all the cells. I believe structure_name can be a placeholder like NA
since none of the internal cell structures are labeled.
"""

import os
import pathlib

import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer

from utils import rotate_image_2d

# Read in manifest to be updated
project_dir = '/home/maddy/projects/claudin_gfp_5dpf_airy_live/cvapipe_run_2/'
path_to_manifest = f'{project_dir}/alignment/manifest.csv'

cell_df = pd.read_csv(path_to_manifest, index_col=0)

# Rename old crop_seg and crop_raw since these were pre-alignment
# And I want to used the aligned single cell masks going forward
cell_df = cell_df.rename(columns={
    'crop_raw': 'crop_raw_old',
    'crop_seg': 'crop_seg_old'
})

# Add name_dict
name_dict = {
        'crop_raw': ['membrane'],
        'crop_seg': ['cell_seg']
}

num_cells = len(cell_df)
name_dict_list = [name_dict] * num_cells
cell_df['name_dict'] = name_dict_list

# Add structure_name
structure_name = 'NA'

cell_df['structure_name'] = structure_name

# We do need the raw images to be rotated too, oops
aligned_paths = []
for row in cell_df.itertuples(index=False):
    reader = AICSImage(row.crop_raw_old)
    raw_img = reader.get_image_data('ZYX', S=0, T=0, C=0)
    angle = float(row.rotation_angle)

    if raw_img.ndim == 3:
        raw_img = np.expand_dims(raw_img, axis=0)
    raw_aligned = rotate_image_2d(
            image=raw_img,
            angle=angle,
            interpolation_order=0
    )

    # Save and update crop_seg and crop_raw columns while we're at it
    current_cell_dir = f'{project_dir}/alignment/{row.fov_id}/{row.label}'
    assert os.path.isdir(current_cell_dir)

    raw_path = f'{current_cell_dir}/raw.ome.tif'
    crop_raw_aligned_path = pathlib.Path(raw_path)
    writer = ome_tiff_writer.OmeTiffWriter(crop_raw_aligned_path)
    writer.save(raw_aligned, dimension_order='CZYX')

    seg_path = f'{current_cell_dir}/segmentation.ome.tif'
    crop_seg_aligned_path = pathlib.Path(seg_path)
    assert crop_seg_aligned_path.exists()

    aligned_paths.append({
        'CellId': row.CellId,
        'crop_raw': raw_path,
        'crop_seg': seg_path
    })

# Update df
path_df = pd.DataFrame(aligned_paths)
cell_df = cell_df.merge(path_df, on='CellId')

# Save updated manifest
cell_df.to_csv(f'{project_dir}/manifest.csv')
