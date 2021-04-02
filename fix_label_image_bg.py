#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import sys

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer

from utils import get_largest_cc, switch_label_values

# Add command line entry points
parser = argparse.ArgumentParser()
parser.add_argument('source', help='source dir of images to process')
parser.add_argument('destination', help='destination for processed images')
parser.add_argument('extension', help='file extension to use, e.g. tiff')
args = parser.parse_args()

input_path = args.source
save_path = args.destination
extension = args.extension

if not os.path.isdir(input_path):
    print('Input path does not exist')
    sys.exit()

if not os.path.isdir(save_path):
    print('Save path does not exist')
    sys.exit()

# Find all files in source dir
list_of_files = glob.glob(f'{input_path}/*.{extension}')

for count, filename in enumerate(list_of_files):
    reader = AICSImage(filename)
    label_img = reader.get_image_data('ZYX', S=0, T=0, C=0)
    lcc = get_largest_cc(label_img)
    label_img = switch_label_values(label_img, 0, lcc)
    basename = os.path.basename(filename)
    writer = ome_tiff_writer.OmeTiffWriter(f'{save_path}/{basename}')
    writer.save(label_img)
