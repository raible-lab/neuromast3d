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

# Check that paths are valid
# TODO: Add other checks?
if not os.path.isdir(raw_source_dir):
    print('Raw source dir does not exist')
    sys.exit()

if not os.path.isdir(seg_source_dir):
    print('Seg source dir does not exist')
    sys.exit()

# Find all files in source dirs and gather image names into list
raw_files = sorted(glob.glob(f'{raw_source_dir}/*.{extension}'))
seg_files = sorted(glob.glob(f'{seg_source_dir}/*.{extension}'))

if not len(raw_files) == len(seg_files):
    print('Number of raw files does not match number of seg files.')
    sys.exit()

raw_img_names = []
for fn in raw_files:
    bn = os.path.basename(fn)
    raw_img_name = bn.split('.')[0]
    raw_img_names = np.append(raw_img_names, raw_img_name)

seg_img_names = []
for fn in seg_files:
    bn = os.path.basename(fn)
    seg_img_name = bn.rpartition('_')[0]

    # Some label images have different suffixes
    # If both 'raw' and 'edited' exist, we only want the 'edited' ones
    if bn.rpartition('_')[2] == 'editedlabels.tiff':
        seg_files.remove(f'{seg_source_dir}/{seg_img_name}_rawlabels.tiff')

# Create fov dataframe (where every row is a neuromast)
fov_dataset = pd.DataFrame({'NM_ID': raw_img_names})
fov_dataset = fov_dataset.sort_values(by=['NM_ID'], ignore_index=True)

# TODO: This is where we left off last time
# Feel like there is a better way of doing this...

# Create single cell dataset from nm label images
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
