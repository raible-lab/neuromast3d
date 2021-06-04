#!usr/bin/env python3
# -*- coding: utf-8 -*-

""" Prepare single cells for cvapipe_analysis

This script will separate label images of neuromasts into single cell images as
binary masks. The cells are interpolated along z, cropped, and centered. This
processing is done to both the label image and the corresponding raw image.

Note: This script is based on the prep_analysis_single_cell_utils.py in the
AICS cvapipe repo. Please see https://github.com/AllenCell/cvapipe for details.
"""

import argparse
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd

from aicsimageio import AICSImage, imread
from aicsimageio.writers import ome_tiff_writer
from aicsimageprocessing import resize, resize_to

logger = logging.getLogger(__name__)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('raw_dir', help='directory containing raw images')
parser.add_argument('seg_dir', help='directory containing segmented images')
parser.add_argument('extension', help='file extension to use, e.g. tiff')
parser.add_argument('dest', help='directory in which to save the files')
parser.add_argument('z_res', help='voxel depth', type=float)
parser.add_argument('xy_res', help='pixel size in xy', type=float)
parser.add_argument(
        '-o',
        '--overwrite',
        help='overwrite files in destination directory',
        action='store_true'
)

args = parser.parse_args()

raw_source_dir = args.raw_dir
seg_source_dir = args.seg_dir
extension = args.extension
dest_dir = args.dest
z_res = args.z_res
xy_res = args.xy_res

# Save command line arguments into logfile
log_file_path = f'{dest_dir}/prep_single_cells.log'
logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s %(message)s'
)
logger.info(sys.argv)

# Check that paths are valid
if not os.path.isdir(raw_source_dir):
    print('Raw source dir does not exist')
    sys.exit()

if not os.path.isdir(seg_source_dir):
    print('Seg source dir does not exist')
    sys.exit()

if not os.path.isdir(dest_dir):
    print('Destination directory does not exist')
    sys.exit()

# Find all files in source dirs and gather image names into list
raw_files = sorted(glob.glob(f'{raw_source_dir}/*.{extension}'))
seg_files = sorted(glob.glob(f'{seg_source_dir}/*.{extension}'))

# Some label images have different suffixes
# If both 'raw' and 'edited' exist, we only want the 'edited' ones
for fn in seg_files:
    bn = os.path.basename(fn)
    seg_img_name = bn.rpartition('_')[0]
    if bn.rpartition('_')[2] == 'editedlabels.tiff':
        seg_files.remove(f'{seg_source_dir}/{seg_img_name}_rawlabels.tiff')

if not len(raw_files) == len(seg_files):
    print('Number of raw files does not match number of seg files.')
    sys.exit()

# Create initial fov dataframe (where every row is a neuromast)
raw_img_names = []
for fn in raw_files:
    bn = os.path.basename(fn)
    raw_img_name = bn.split('.')[0]
    raw_img_names.append({'NM_ID': raw_img_name, 'SourceReadPath': fn})

seg_img_names = []
for fn in seg_files:
    bn = os.path.basename(fn)
    seg_img_name = bn.rpartition('_')[0]
    seg_img_names.append(
            {
                'NM_ID': seg_img_name,
                'MembraneSegmentationReadPath': fn
            }
    )

raw_df = pd.DataFrame(raw_img_names)
seg_df = pd.DataFrame(seg_img_names)
fov_dataset = raw_df.merge(seg_df, on='NM_ID')
fov_dataset.to_csv(f'{dest_dir}/fov_dataset.csv')

# TODO: should fov_dataset generation be its own script or function?

# TODO: add single cell QC, for example remove cells based on size threshold
# Min/max size could be passed via command line

# Create dir for single cell masks to go into
if not os.path.exists(f'{dest_dir}/single_cell_masks'):
    os.mkdir(f'{dest_dir}/single_cell_masks')
elif os.path.exists(f'{dest_dir}/single_cell_masks'):
    print('single cell masks dir already exists')

# Actual cell dataset creation here (above chunk can maybe be deleted?)
cell_meta = []
for row in fov_dataset.itertuples(index=False):
    current_fov_dir = f'{dest_dir}/single_cell_masks/{row.NM_ID}'
    if not os.path.exists(current_fov_dir):
        os.mkdir(current_fov_dir)
    reader_raw = AICSImage(row.SourceReadPath)
    raw_img = reader_raw.get_image_data('ZYX', S=0, T=0, C=0)
    reader_seg = AICSImage(row.MembraneSegmentationReadPath)
    seg_img = reader_seg.get_image_data('ZYX', S=0, T=0, C=0)
    raw_img_rescaled = resize(
            raw_img,
            (z_res / xy_res, xy_res / xy_res, xy_res / xy_res),
            method='bilinear'
    ).astype(np.uint16)
    raw_img_whole = resize_to(
            raw_img, raw_img_rescaled.shape, method='nearest'
    )
    mem_seg_whole = resize_to(
            seg_img, raw_img_rescaled.shape, method='nearest'
    )
    cell_label_list = list(np.unique(mem_seg_whole[mem_seg_whole > 0]))
    for label in cell_label_list:
        if not os.path.exists(f'{current_fov_dir}/{label}'):
            os.mkdir(f'{current_fov_dir}/{label}')
        mem_seg = mem_seg_whole == label
        z_range = np.where(np.any(mem_seg, axis=(1, 2)))
        y_range = np.where(np.any(mem_seg, axis=(0, 2)))
        x_range = np.where(np.any(mem_seg, axis=(0, 1)))
        z_range = z_range[0]
        y_range = y_range[0]
        x_range = x_range[0]
        roi = [
                max(z_range[0] - 10, 0),
                min(z_range[-1] + 12, mem_seg.shape[0]),  # not sure why 12
                max(y_range[0] - 40, 0),
                min(y_range[-1] + 40, mem_seg.shape[1]),
                max(x_range[0] - 40, 0),
                min(x_range[-1] + 40, mem_seg.shape[2])
        ]
        mem_seg = mem_seg.astype(np.uint8)
        mem_seg = mem_seg[roi[0]: roi[1], roi[2]: roi[3], roi[4]: roi[5]]
        mem_seg[mem_seg > 0] = 255
        crop_seg_path = os.path.join(
                f'{current_fov_dir}/{label}',
                'segmentation.ome.tif'
        )
        writer = ome_tiff_writer.OmeTiffWriter(crop_seg_path)
        writer.save(mem_seg, dimension_order='ZYX')
        raw_img = raw_img_whole[roi[0]: roi[1], roi[2]: roi[3], roi[4]: roi[5]]
        crop_raw_path = os.path.join(
                f'{current_fov_dir}/{label}',
                'raw.ome.tif'
        )
        writer = ome_tiff_writer.OmeTiffWriter(crop_raw_path)
        writer.save(raw_img, dimension_order='ZYX')
        cell_id = f'{row.NM_ID}_{label}'

        # Add name dict
        # NOTE: currently hardcoded. may need to change depending on run
        name_dict = {
                'crop_raw': ['membrane'],
                'crop_seg': ['cell_seg'],
        }

        # If no structure name, need 'NA' as a placeholder for future steps
        if 'Gene' in fov_dataset:
            structure_name = row.Gene
        else:
            structure_name = 'NA'
        cell_meta.append(
                {
                    'CellId': cell_id,
                    'label': label,
                    'roi': roi,
                    'crop_raw': crop_raw_path,
                    'crop_seg': crop_seg_path,
                    'scale_micron': [xy_res, xy_res, xy_res],
                    'fov_id': row.NM_ID,
                    'fov_path': row.SourceReadPath,
                    'fov_seg_path': row.MembraneSegmentationReadPath,
                    'name_dict': name_dict,
                    'structure_name': structure_name
                }
        )

# Save cell dataset (every row is a cell)
df_cell_meta = pd.DataFrame(cell_meta)
df_cell_meta.to_csv(f'{dest_dir}/cell_manifest.csv')
