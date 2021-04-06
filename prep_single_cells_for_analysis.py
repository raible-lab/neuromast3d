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
import os
import sys

import numpy as np
import pandas as pd

from aicsimageio import AICSImage, imread
from aicsimageio.writers import ome_tiff_writer
from aicsimageprocessing import resize, resize_to

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('raw_dir', help='directory containing raw images')
parser.add_argument('seg_dir', help='directory containing segmented images')
parser.add_argument('extension', help='file extension to use, e.g. tiff')
parser.add_argument('dest', help='directory in which to save the files')
parser.add_argument(
        'voxel_size',
        help='3-tuple corresponding to pixel size in X, Y, and Z respectively'
)
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
PixelSizeX = args.voxel_size[0]
PixelSizeY = args.voxel_size[1]
PixelSizeZ = args.voxel_size[2]

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
    seg_img_names.append({
        'NM_ID': seg_img_name, 'MembraneSegmentationReadPath': fn
    })

raw_df = pd.DataFrame(raw_img_names)
seg_df = pd.DataFrame(seg_img_names)
fov_dataset = raw_df(seg_df, on='NM_ID')
fov_dataset.to_csv(f'{dest_dir}/fov_dataset.csv')

# TODO: Check that new way of making initial fov_dataset works

# Create unique cell_ids
nm_id_list = []
cell_id_list = []
for seg_fn in seg_files:
    nm_id = bn.rpartition('_')[0]
    reader = AICSImage(seg_fn)
    mem_seg_whole = reader.get_image_data('ZYX', S=0, T=0, C=0)
    cell_label_list = list(np.unique(mem_seg_whole[mem_seg_whole > 0]))
    for label in cell_label_list:
        cell_id = f'{nm_id}_{label}'
        cell_id_list = np.append(cell_id_list, cell_id)
        nm_id_list = np.append(nm_id_list, nm_id)
    break

# TODO: add single cell QC, for example remove cells based on size threshold
# Min/max size could be passed via command line

cell_dataset = pd.DataFrame({'NM_ID': nm_id_list, 'CellID': cell_id_list})

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
