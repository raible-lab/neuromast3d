#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Segment binarized nuclei using distance transform watershed

Note: This scripts assumes a membrane channel is available as well.
"""

import argparse
import logging
import os
from pathlib import Path
import sys

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
import napari
import numpy as np
from skimage import morphology, filters

from neuromast3d.segmentation.utils import dt_watershed


# Command line arguments
parser = argparse.ArgumentParser(
        description='DT watershed segmentation of nucleus mask'
)
parser.add_argument('raw_dir', help='directory containing raw input images')
parser.add_argument('mask_dir', help='directory containing nuclear masks')
parser.add_argument('output_dir', help='directory to save output labels')
parser.add_argument('sigma', type=int, help='sigma to use for Gaussian blur of \
        distance transform')
parser.add_argument('min_distance', type=int, help='min_distance parameter for \
        peak_local_max function')

# Parse and save as variables
args = parser.parse_args()
raw_dir = args.raw_dir
mask_dir = args.mask_dir
output_dir = args.output_dir

# Check raw and mask directories exist
if not os.path.isdir(raw_dir):
    print('Raw directory does not exist')
    sys.exit()

if not os.path.isdir(mask_dir):
    print('Mask directory does not exist')
    sys.exit()

# Create output directory
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Save command line arguments into log file
logger = logging.getLogger(__name__)
log_file_path = output_dir / 'nuc_seg.log'
logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s %(message)s'
)
logger.info(sys.argv)

# Collect all image ids in raw directory
list_of_img_ids = [fn.stem for fn in Path(raw_dir).glob('*.tiff')]

# Loop over matching raw and mask images
for img_id in list_of_img_ids:

    # Check that edited labels haven't already been created
    final_output_path = Path(
            f'{output_dir}/edited_nuc_labels/{img_id}_editedlabels.tiff'
    )

    # If it does, go to next image
    if final_output_path.exists():
        continue

    # Read raw image
    path = Path(f'{raw_dir}/{img_id}.tiff')
    reader = AICSImage(path)
    raw_img = reader.get_image_data('CZYX', S=0, T=0)

    # Read mask image
    path = Path(f'{mask_dir}/{img_id}_nuc_seg.tiff')
    reader = AICSImage(path)
    nuc_mask = reader.get_image_data('ZYX', C=0, S=0, T=0)

    # Split raw into membrane and nucleus channels
    # Note: Could refactor to allow passing channel index as arg
    # In case this varies
    membranes = raw_img[0, :, :, :]
    nuclei = raw_img[1, :, :, :]

    # Blur, threshold, and erode the membrane image
    mem_blurred = filters.gaussian(membranes, sigma=1, preserve_range=True)
    thresh_otsu = filters.thresholding.threshold_otsu(membranes)
    mem_otsu = mem_blurred > thresh_otsu
    mem_eroded = morphology.binary_erosion(mem_otsu)

    # Use eroded membrane mask to split touching nuclei
    nuclei_split = np.where(mem_eroded, 0, nuc_mask)

    # Apply dt watershed
    ws_results = dt_watershed(
            nuclei_split,
            sigma=args.sigma,
            min_distance=args.min_distance
    )

    # Save raw (unedited) labels
    raw_save_path = Path(f'{output_dir}/raw_nuc_labels')
    raw_save_path.mkdir(parents=True, exist_ok=True)
    writer = ome_tiff_writer.OmeTiffWriter(
            raw_save_path/f'{img_id}_rawlabels.tiff',
            overwrite_file=True
    )
    writer.save(ws_results, dimension_order='ZYX')
    logger.info('%s raw labels saved at %s', img_id, raw_save_path)

    # Inspect the results in the napari viewer
    viewer = napari.Viewer()
    viewer.add_image(nuclei, colormap='gray', blending='additive')
    viewer.add_image(membranes, colormap='cyan', blending='additive')
    viewer.add_image(nuc_mask, colormap='magenta', blending='additive')
    viewer.add_image(nuclei_split, colormap='green', blending='additive')
    label_layer = viewer.add_labels(ws_results)
    napari.run()

    # Save edited labels from napari
    edited_labels = label_layer.data
    edited_save_path = Path(f'{output_dir}/edited_nuc_labels')
    edited_save_path.mkdir(parents=True, exist_ok=True)
    writer = ome_tiff_writer.OmeTiffWriter(
            edited_save_path/f'{img_id}_editedlabels.tiff'
    )
    writer.save(edited_labels, dimension_order='ZYX')
    logger.info('%s edited labels saved at %s', img_id, edited_save_path)

    # Check if the user wants to continue annotating the next image
    # If not, next time the script is run, it will use the logfile
    # To determine which image to start with
    user_response = input(f'Edited labels for {img_id} saved. \
            Would you like to continue with the next image? (y/n)')
    if user_response == 'y':
        print('Continuing with next image.')
        continue
    elif user_response == 'n':
        print('Terminating annotations. Run this script again to resume.')
        break
    else:
        print('Invalid input. Please enter y to continue or n to exit.')
