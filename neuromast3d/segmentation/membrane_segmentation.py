#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Interactive membrane segmentation script

Takes as input: dual-channel images with nucleus and membrane labeled
Also probability map for cell boundaries from DL model

Output: cell instance segmentation labels
"""

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import List, Tuple, Union

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
from magicgui import magicgui
import napari
from napari.layers import Image, Labels, Layer, Points
from napari.types import LabelsData
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, filters
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed


# Initialize napari viewer
viewer = napari.Viewer()


# Add custom keybindings, functions, etc. to napari viewer
@viewer.bind_key('s')
def save_labels_layer(viewer):
    '''Save current mem labels during annotation to temp directory'''
    if viewer.layers('mem_labels'):
        mem_labels_temp = viewer.layers('mem_labels').data
        temp_dir = output_dir / 'temp_mem_labels'
        reader = ome_tiff_writer.OmeTiffWriter(temp_dir, overwrite_file=True)
        reader.save(mem_labels_temp)


@magicgui(call_button='Clear current viewer')
def clear_layers(layer: Layer):
    if viewer.layers:
        viewer.layers.clear()


raw_dir = Path('/home/maddy/projects/claudin_gfp_5dpf_DRAQ5_1h_airy_live/stack_aligned')
nuc_labels_dir = Path('/home/maddy/projects/claudin_gfp_5dpf_DRAQ5_1h_airy_live/labels/edited_nuc_labels')
mem_labels_dir = Path('/home/maddy/projects/claudin_gfp_5dpf_DRAQ5_1h_airy_live/claudin_model_iter2_results')
list_of_img_ids = [fn.stem for fn in Path(raw_dir).glob('*.tiff')]


@magicgui(call_button='Open next image', img_id={'choices': list_of_img_ids})
def open_next_image(img_id) -> List[napari.types.LayerDataTuple]:
    reader = AICSImage(f'{raw_dir}/{img_id}.tiff')
    image = reader.get_image_data('CZYX', S=0, T=0)

    reader = AICSImage(f'{nuc_labels_dir}/{img_id}_editedlabels.tiff')
    nuc_labels = reader.get_image_data('ZYX', C=0, S=0, T=0)

    reader = AICSImage(f'{mem_labels_dir}/{img_id}_struct_segmentation.tiff')
    mem_labels = reader.get_image_data('ZYX', C=0, S=0, T=0)
    return [(image, {'name': 'raw'}, 'image'),
            (nuc_labels, {'name': 'nuc_labels'}, 'labels'),
            (mem_labels, {'name': 'mem_labels'}, 'image')]


@magicgui(call_button='Generate seeds')
def generate_seeds_from_nuclei(nuc_labels_layer: LabelsData) -> Points:
    if viewer.layers['nuc_labels']:
        seeds = []
        for region in regionprops(nuc_labels_layer):
            seeds.append(np.round(region['centroid']))

        return Points(seeds)


viewer.window.add_dock_widget(clear_layers)
viewer.window.add_dock_widget(open_next_image)
viewer.window.add_dock_widget(generate_seeds_from_nuclei)

napari.run()
'''
# Command line arguments
parser = argparse.ArgumentParser(
        description='Interactive membrane segmentation script'
)
parser.add_argument('raw_dir', help='directory containing raw imput images')
parser.add_argument('nuc_labels_dir')
parser.add_argument('mem_prob_map_dir')
parser.add_argument('output_dir')

# Parse args and save as variables
args = parser.parse_args()
raw_dir = args.raw_dir
nuc_labels_dir = args.nuc_labels_dir
mem_prob_map_dir = args.mem_prob_map_dir
output_dir = args.output_dir

# Check raw and nuc labels dir exist
if not os.path.isdir(raw_dir):
    print('Raw directory does not exist')
    sys.exit()

if not os.path.isdir(nuc_labels_dir):
    print('Nucleus labels directory does not exist')
    sys.exit()

if not os.path.isdir(mem_prob_map_dir):
    print('Membrane probability map dir does not exist')
    sys.exit()

# Create output directory
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Save command line arguments to log file
logger = logging.getLogger(__name__)
log_file_path = output_dir / 'mem_seg.log'
logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s %(message)s'
)
logger.info(sys.argv)

# Collect all image ids in raw directory
list_of_img_ids = [fn.stem for fn in Path(raw_dir).glob('*.tiff')]

napari.run()
'''
