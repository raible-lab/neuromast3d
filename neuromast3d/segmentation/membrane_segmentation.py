#!/usr/bin/env python3
#-*- coding: utf-8 -*-

""" Semi-automated membrane segmentation script """

import argparse
import logging
from pathlib import Path
import sys

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
from magicgui import magicgui
import napari
import numpy as np

from neuromast3d.segmentation.utils import dt_watershed

# Command line arguments
parser = argparse.ArgumentParser(
        description='Semi-automated membrane segmentation'
)
parser.add_argument('raw_dir', help='directory containing raw input images')
parser.add_argument('nuc_labels_dir', help='directory with segmented nuclei')
parser.add_argument('mem_pred_dir', help='directory with mem boundary predictions')
parser.add_argument('output_dir', help='directory to save membrane labels')

# Parse and save as variables
args = parser.parse_args()
raw_dir = args.raw_dir
nuc_labels_dir = args.nuc_labels_dir
mem_pred_dir = args.mem_pred_dir
output_dir = args.output_dir

# Create output directory if it doesn't already exist
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Save command line arguments to log file and output to console
logger = logging.getLogger(__name__)
log_file_path = output_dir / 'mem_seg.log'
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
)
logger.info(sys.argv)


