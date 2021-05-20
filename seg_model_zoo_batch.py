#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Simple script to apply AICS pretrained models on a directory of images """

import argparse
import logging
from pathlib import Path
import sys

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
from aicsmlsegment.utils import background_sub, simple_norm
import numpy as np
from segmenter_model_zoo.zoo import SegModel


# Command line arguments
parser = argparse.ArgumentParser(
        description='apply AICS pretrained model on a dir of images \
                to generate binary masks.'
)
parser.add_argument('input_dir', help='directory containing input images')
parser.add_argument('output_dir', help='directory in which to save masks')
parser.add_argument('model_name', type=str, help='name of pretrained model')
parser.add_argument('normalization_recipe', type=int, nargs=3,
                    help='custom recipe for image normalization. Provide as'
                    'three numbers corresponding to bg_sub radius,'
                    'simple norm lower value, and simple norm higher'
                    'value. Example: 50 1 5')
parser.add_argument('input_ch', type=int, nargs=1,
                    help='channel on which to apply the model')

args = parser.parse_args()
input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
normalization_vals = args.normalization_recipe
backsub_radius = int(normalization_vals[0])
simple_norm_lower = int(normalization_vals[1])
simple_norm_upper = int(normalization_vals[2])
input_ch = args.input_ch

# Check input dir exists
if not input_dir.is_dir():
    print('Input dir does not exist')
    sys.exit()

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Save command line arguments to log file
logger = logging.getLogger(__name__)
log_file_path = output_dir / 'model_inference_params.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO)
logger.info(sys.argv)

# Load pretrained model
model = SegModel()
model.load_train('H2B_coarse', {'local_path': './aics_seg_model_zoo_models'})
model.to_gpu('cuda:0')

# Apply model to images and save output binary masks
list_of_files = Path(input_dir).glob('*.tiff')
for path in list_of_files:
    reader = AICSImage(path)
    dna_img = reader.get_image_data(
            'CZYX', C=input_ch, S=0, T=0
    ).astype(np.float32)
    dna_image_backsub = background_sub(dna_img, backsub_radius)
    dna_image_normalized = simple_norm(
            dna_image_backsub,
            simple_norm_lower,
            simple_norm_upper
    )
    nuclear_mask_from_dye = model.apply_on_single_zstack(
            dna_image_normalized,
            already_normalized=True
    )
    img_name = path.stem
    writer = ome_tiff_writer.OmeTiffWriter(
            output_dir / f'{img_name}_nuc_seg.tiff'
    )
    writer.save(nuclear_mask_from_dye, dimension_order='ZYX')
