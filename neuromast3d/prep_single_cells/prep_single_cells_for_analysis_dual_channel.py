#!usr/bin/env python3
# -*- coding: utf-8 -*-

""" Prepare single cells for cvapipe_analysis (dual channel version) """

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
from aicsimageprocessing import resize, resize_to

logger = logging.getLogger(__name__)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('raw_dir', help='directory containing raw images')
parser.add_argument('seg_dir', help='directroy contianing segmented image')
parser.add_argument('extension', help='file extension to use, e.g. tiff')
parser.add_argument('dest', help='directory in which to save output files')
parser.add_argument('z_res', help='voxel depth', type=float)
parser.add_argument('xy_res', help='pixel size in xy', type=float)
parser.add_argument('raw_nuc_ch_index', type=int)
parser.add_argument('raw_mem_ch_index', type=int)
parser.add_argument('seg_nuc_ch_index', type=int)
parser.add_argument('seg_mem_ch_index', type=int)

# Parse arguments
args = parser.parse_args()
raw_dir = Path(args.raw_dir)
seg_dir = Path(args.seg_dir)
extension = args.extension
dest_dir = Path(args.dest)
z_res = args.z_res
xy_res = args.xy_res

# Create destination directory if it doesn't exist
dest_dir.mkdir(parents=True, exist_ok=True)

# Save command line arguments into logfile
log_file_path = dest_dir / 'prep_single_cells.log'
logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s %(message)s'
)
logger.info(sys.argv)

# Check paths exist
if not raw_dir.is_dir():
    print('Raw dir does not exist')
    sys.exit()

if not seg_dir.is_dir():
    print('Seg dir does not exist')
    sys.exit()

# Find all files in source directroes and gather images filenames into list
raw_files = [fn for fn in raw_dir.glob(f'*.{extension}')]
seg_files = [fn for fn in seg_dir.glob(f'*.{extension}')]

if not len(raw_files) == len(seg_files):
    print('Number of raw files does not match number of seg files.')
    sys.exit()

# Create initial fov dataframe (where every row is a neuromast)
raw_img_names = []
for fn in raw_files:
    raw_img_name = fn.stem
    raw_img_names.append(
            {
                'NM_ID': raw_img_name,
                'SourceReadPath': fn,
                'RawNucChannelIndex': args.raw_nuc_ch_index,
                'RawMemChannelIndex': args.raw_mem_ch_index
            }
    )

seg_img_names = []
for fn in seg_files:
    seg_img_name = fn.stem

    # Segmented images usually have some suffix after a '_'
    seg_img_name = seg_img_name.rpartition('_')[0]
    seg_img_names.append(
            {
                'NM_ID': seg_img_name,
                'SegmentationReadPath': fn,
                'SegNucChannelIndex': args.seg_nuc_ch_index,
                'SegMemChannelIndex': args.seg_mem_ch_index
            }
    )

# Create fov dataframe from dicts and save
raw_df = pd.DataFrame(raw_img_names)
seg_df = pd.DataFrame(seg_img_names)
fov_dataset = raw_df.merge(seg_df, on='NM_ID')
# fov_dataset.to_csv(f'{dest_dir}/fov_dataset.csv)

# Create dir for single cells to go into
single_cell_dir = dest_dir / 'single_cell_masks'
single_cell_dir.mkdir(parents=True, exist_ok=True)

# Create the cell dataset
# Iterate through cells in each FOV
cell_meta = []
for row in fov_dataset.itertuples(index=False):

    # Make dir for this FOV
    current_fov_dir = single_cell_dir / f'{row.NM_ID}'
    current_fov_dir.mkdir(parents=True, exist_ok=True)

    # Get the raw and segmented FOV images
    # Note: channels for mem and nuc are currently hardcoded
    reader_raw = AICSImage(row.SourceReadPath)
    mem_raw = reader_raw.get_image_data('ZYX', S=0, T=0, C=row.RawMemChannelIndex)
    nuc_raw = reader_raw.get_image_data('ZYX', S=0, T=0, C=row.RawNucChannelIndex)

    reader_seg = AICSImage(row.SegmentationReadPath)
    mem_seg = reader_seg.get_image_data('ZYX', S=0, T=0, C=row.SegMemChannelIndex)
    nuc_seg = reader_seg.get_image_data('ZYX', S=0, T=0, C=row.SegNucChannelIndex)

    # Discard nuclei not included in the membrane
    # NOTE: make sure this part works as expected
    dna_mask_bw = nuc_seg
    dna_mask_bw[mem_seg == 0] = 0

    # Make nuclei labels match the cell's
    dna_mask_label = np.zeros_like(mem_seg)
    dna_mask_label[dna_mask_bw > 0] = 1
    dna_mask_label = dna_mask_label * mem_seg
    nuc_seg = dna_mask_label

    # Rescale raw images to isotropic dimenstions
    raw_nuc_whole = resize(
            nuc_raw,
            (z_res / xy_res, xy_res / xy_res, xy_res / xy_res),
            method='bilinear'
    ).astype(np.uint16)

    raw_mem_whole = resize(
            mem_raw,
            (z_res / xy_res, xy_res / xy_res, xy_res / xy_res),
            method='bilinear'
    ).astype(np.uint16)

    # Resize seg images to match
    mem_seg_whole = resize_to(mem_seg, raw_mem_whole.shape, method='nearest')
    nuc_seg_whole = resize_to(nuc_seg, raw_nuc_whole.shape, method='nearest')

    # Remove very small cells from the list
    cell_label_list = list(np.unique(mem_seg_whole[mem_seg_whole > 0]))
    cell_label_list_copy = cell_label_list.copy()
    for count, label in enumerate(cell_label_list):
        single_cell_mem = nuc_seg_whole == label
        single_cell_nuc = mem_seg_whole == label

        # These numbers are guessed as reasonable thresholds
        if(
                np.count_nonzero(single_cell_mem) < 1000
                or np.count_nonzero(single_cell_nuc) < 100
        ):
            cell_label_list_copy.remove(label)

        # Could add other QC, but let's just try this for now

    # A bit awkward, but we can't remove items from a list we're iterating over
    cell_label_list = cell_label_list_copy.copy()

    # Crop and prep the cells
    # NOTE: this part copy/pasted, double check
    for label in cell_label_list:
        label_dir = current_fov_dir / f'{label}'
        label_dir.mkdir(parents=True, exist_ok=True)
        mem_seg = mem_seg_whole == label
        nuc_seg = nuc_seg_whole == label

        # Make the cropping roi
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

        # Crop both segmentation channels to membrane roi
        mem_seg = mem_seg.astype(np.uint8)
        mem_seg = mem_seg[roi[0]: roi[1], roi[2]: roi[3], roi[4]: roi[5]]
        mem_seg[mem_seg > 0] = 255

        nuc_seg = nuc_seg.astype(np.uint8)
        nuc_seg = nuc_seg[roi[0]: roi[1], roi[2]: roi[3], roi[4]: roi[5]]
        nuc_seg[nuc_seg > 0] = 255

        # Merge channels and save
        mem_seg = np.expand_dims(mem_seg, axis=0)
        nuc_seg = np.expand_dims(nuc_seg, axis=0)
        seg_merged = np.concatenate([nuc_seg, mem_seg], axis=0)

        crop_seg_path = label_dir / 'segmentation.ome.tif'
        writer = ome_tiff_writer.OmeTiffWriter(crop_seg_path)
        writer.save(seg_merged, dimension_order='CZYX')

        # Crop both channels of the raw image
        raw_nuc = raw_nuc_whole[roi[0]: roi[1], roi[2]: roi[3], roi[4]: roi[5]]
        raw_mem = raw_mem_whole[roi[0]: roi[1], roi[2]: roi[3], roi[4]: roi[5]]

        # Merge channels and save
        raw_nuc = np.expand_dims(raw_nuc, axis=0)
        raw_mem = np.expand_dims(raw_mem, axis=0)
        raw_merged = np.concatenate([raw_nuc, raw_mem], axis=0)

        crop_raw_path = label_dir / 'raw.ome.tif'
        writer = ome_tiff_writer.OmeTiffWriter(crop_raw_path)
        writer.save(raw_merged, dimension_order='CZYX')
        cell_id = f'{row.NM_ID}_{label}'

        # Add name dict
        # NOTE: currently hardcoded. may need to change depending on run
        name_dict = {
                 'crop_raw': ['dna', 'membrane'],
                 'crop_seg': ['dna_seg', 'cell_seg'],
         }

        # If no structure name, add NoStr as a placeholder for future steps
        # Note: avoid using NA or other pandas default_na_values
        if 'Gene' in fov_dataset:
            structure_name = row.Gene
        else:
            structure_name = 'NoStr'
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
                     'fov_seg_path': row.SegmentationReadPath,
                     'name_dict': name_dict,
                     'structure_name': structure_name,
                     'RawNucChannelIndex': row.RawNucChannelIndex,
                     'RawMemChannelIndex': row.RawMemChannelIndex,
                     'SegNucChannelIndex': row.SegNucChannelIndex,
                     'SegMemChannelIndex': row.SegMemChannelIndex
                }
        )

# Save cell dataset (every row is a cell)
df_cell_meta = pd.DataFrame(cell_meta)
path_to_manifest = dest_dir / 'cell_mainfest.csv'
df_cell_meta.to_csv(path_to_manifest)
