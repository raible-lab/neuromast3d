#!usr/bin/env python3
# -*- coding: utf-8 -*-

""" Prepare a fov dataset (without splitting into single cells

Allows addition of manual values to the FOV dataset to use for tilt correction
etc.
"""

import argparse
import logging
from pathlib import Path
import re
import sys

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
from magicgui import magicgui
import napari
from napari.types import ImageData, LabelsData
import numpy as np
import pandas as pd
from scipy.ndimage import rotate


logger = logging.getLogger(__name__)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('original_dir', help='directory containing OG images')
parser.add_argument('raw_dir', help='directory containing raw images')
parser.add_argument('seg_dir', help='directory containing segmented images')
parser.add_argument('output_dir', help='directory in which to save outputs')

# Parse arguments
args = parser.parse_args()
original_dir = Path(args.original_dir)
raw_dir = Path(args.raw_dir)
seg_dir = Path(args.seg_dir)
output_dir = Path(args.output_dir)

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Save command line arguments into logfile
log_file_path = output_dir / 'create_fov_dataset.log'
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

# Find all files in source directories and gather image filenames into list
raw_files = list(raw_dir.glob('*.tiff'))
seg_files = list(seg_dir.glob('*.tiff'))
og_files = list(original_dir.glob('*.czi'))

if not len(raw_files) == len(seg_files):
    print('Number of raw files does not match number of seg files.')
    sys.exit()

# Create initial fov dataframe (where every row is a neuromast)
# There's probably a better way to do this that doesn't involve two for loops
# But I am working on something else right now, so - TODO?
# Lol nevermind I figured it out, but now I forgot what I was originally doing

fov_info = []
for fn in raw_files:
    raw_img_name = fn.stem
    pattern = re.compile(raw_img_name)
    seg_img_path = [fn for fn in seg_files if pattern.match(fn.stem)]
    og_img_path = [fn for fn in og_files if pattern.match(fn.stem)]

    try:
        reader = AICSImage(og_img_path[0])
        pixel_size = reader.get_physical_pixel_size()
    except FileNotFoundError:
        pass

    fov_info.append(
            {
                'NM_ID': raw_img_name,
                'SourceReadPath': fn,
                'SegmentationReadPath': seg_img_path,
                'pixel_size_xyz': pixel_size
            }
    )

fov_dataset = pd.DataFrame(fov_info)
fov_dataset.to_csv(output_dir / 'fov_dataset.csv')

# Use napari to find good manual rotation parameters
# Initialize napari viewer
viewer = napari.Viewer()

# TODO: change this so we iterate through all the files
reader = AICSImage(fov_dataset['SourceReadPath'][0])
raw_image = reader.get_image_data('ZYX', C=0, S=0, T=0)
viewer.add_image(raw_image)


# Function to explore rotation parameters
@magicgui(
        auto_call=True,
        angle={'widget_type': 'FloatSlider', 'max': 360},
        mode={'choices': ['reflect', 'constant', 'nearest', 'mirror']},
        layout='horizontal'
)
def rotate_image_interactively(layer: ImageData, angle: float = 0, mode='nearest', order: int = 0) -> ImageData:
    if layer is not None:
        rotated_image = rotate(input=layer, angle=angle, axes=(2, 1), mode=mode, order=order, reshape=True)
        return rotated_image


viewer.window.add_dock_widget(rotate_image_interactively)

napari.run()
