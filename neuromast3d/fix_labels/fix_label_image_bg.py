#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Fix label image background value

This script will fix label images where the background has a non-zero value and
an object (e.g. a cell) has the zero value instead. It assumes that the largest
connected component in the image is the background. It will set the background
to zero and assign the formerly-zero label with the nonzero label that belonged
to the background.

This script accepts label images where each connected component corresponding
to an object has a unique integer value. It accepts any file extensions that
can be read by the AICSImage reader of the aicsimageio library, such as TIFF
and OME-TIFF.
"""

import argparse
import glob
import os
import sys

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer

from utils import get_largest_cc, switch_label_values

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('source', help='source dir of images to process')
parser.add_argument('destination', help='destination for processed images')
parser.add_argument('extension', help='file extension to use, e.g. tiff')
parser.add_argument(
        '-o',
        '--overwrite',
        help='overwrite files in dest',
        action='store_true'
)
args = parser.parse_args()

input_path = args.source
save_path = args.destination
extension = args.extension

# Check that paths passed are valid
if not os.path.isdir(input_path):
    print('Input path does not exist')
    sys.exit()

if not os.path.isdir(save_path):
    print('Save path does not exist')
    sys.exit()

# Find all files in source dir
list_of_files = glob.glob(f'{input_path}/*.{extension}')

# Fix  labels and save
for count, filename in enumerate(list_of_files):
    reader = AICSImage(filename)
    label_img = reader.get_image_data('ZYX', S=0, T=0, C=0)
    lcc = get_largest_cc(label_img)
    label_img = switch_label_values(label_img, 0, lcc)
    basename = os.path.basename(filename)

    if args.overwrite:
        writer = ome_tiff_writer.OmeTiffWriter(
                f'{save_path}/{basename}',
                overwrite_file=True
        )

    else:
        writer = ome_tiff_writer.OmeTiffWriter(
                f'{save_path}/{basename}'
        )

    writer.save(label_img, dimension_order='ZYX')
