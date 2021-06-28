#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Interactive membrane segmentation script

Takes as input: dual-channel images with nucleus and membrane labeled
Also probability map for cell boundaries from DL model

Output: cell instance segmentation labels
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Union

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
from magicgui import magicgui
import napari
from napari.layers import Image, Labels, Points
from napari.types import LabelsData
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, filters
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed


# Command line arguments
parser = argparse.ArgumentParser(
        description='Interactive membrane/nucleus segmentation'
)
parser.add_argument('raw_dir', help='directory containing raw imput images')
parser.add_argument('nuc_labels_dir',
                    help='directory containing segmented nuclei')
parser.add_argument('mem_pred_dir',
                    help='directory containing membrane boundary predictions')
parser.add_argument('output_dir', help='directory to save output images')

# Parse and save as variables
args = parser.parse_args()
raw_dir = args.raw_dir
nuc_labels_dir = args.nuc_labels_dir
mem_pred_dir = args.mem_pred_dir
output_dir = args.output_dir

# Create output directory
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Save command line arguments into log file and output to console
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

# Gather list of img ids
list_of_img_ids = [fn.stem for fn in Path(raw_dir).glob('*.tiff')]

# Initialize napari viewer
viewer = napari.Viewer()


@magicgui(call_button='Clear current viewer')
def clear_layers():
    '''Clear layers in the current viewer.'''
    if viewer.layers:
        viewer.layers.clear()
    return


@magicgui(call_button='Open next image', img_id={'choices': list_of_img_ids})
def open_next_image(img_id) -> List[napari.types.LayerDataTuple]:
    '''Opens corresponding raw image, nuclear labels, and mem predictions'''

    # Open raw image
    reader = AICSImage(f'{raw_dir}/{img_id}.tiff')
    image = reader.get_image_data('CZYX', S=0, T=0)

    # Open nuclear labels
    reader = AICSImage(f'{nuc_labels_dir}/{img_id}_editedlabels.tiff')
    nuc_labels = reader.get_image_data('ZYX', C=0, S=0, T=0)

    # Open mem predictions
    reader = AICSImage(f'{mem_pred_dir}/{img_id}_struct_segmentation.tiff')
    mem_labels = reader.get_image_data('ZYX', C=0, S=0, T=0)

    return [(image, {'name': 'raw', 'blending': 'additive'}, 'image'),
            (nuc_labels, {'name': 'nuc_labels'}, 'labels'),
            (mem_labels, {'name': 'mem_predictions', 'blending': 'additive'}, 'image')]


@magicgui(call_button='Save result to output dir')
def save_layer(
        labels_layer: Labels,
        img_id: Union[str, None] = open_next_image.img_id.value,
        finished: bool = False
):
    '''Save selected layer to output directory.'''

    if img_id is not None:

        if finished:

            # Save to output dir and log this id as finished
            save_path = output_dir / f'{img_id}_{labels_layer.name}_finished.tiff'
            writer = ome_tiff_writer.OmeTiffWriter(save_path)
            writer.save(labels_layer.data, dimension_order='ZYX')
            logger.info(
                    '%s final results for %s saved at %s',
                    img_id,
                    labels_layer.name,
                    save_path
            )

        else:

            # Save temporary results, overwriting any already in the output_dir
            save_path = output_dir / f'{img_id}_{labels_layer.name}_temp.tiff'
            writer = ome_tiff_writer.OmeTiffWriter(save_path, overwrite_file=True)
            writer.save(labels_layer.data, dimension_order='ZYX')
            logger.info(
                    '%s temp results for %s saved at %s',
                    img_id,
                    labels_layer.name,
                    save_path
            )
    return


@magicgui(call_button='Generate seeds')
def generate_seeds_from_nuclei(nuc_labels_layer: LabelsData) -> Points:
    '''Generate one seed for every label, using the centroid of each label'''
    if viewer.layers['nuc_labels']:
        seeds = []
        for region in regionprops(nuc_labels_layer):
            seeds.append(np.round(region['centroid']))

        return Points(seeds)


@magicgui(call_button='Run seeded watershed')
def run_seeded_watershed(
        boundaries: Image,
        seed_dilation_radius: int = 5
) -> Labels:
    '''Run watershed using passed Points layer as seeds.'''
    if viewer.layers['Points']:
        seeds = viewer.layers['Points'].data
        mem_pred_data = boundaries.data
        mask = np.zeros(mem_pred_data.shape, dtype=bool)
        mask[tuple(seeds.astype(int).T)] = True
        markers, _ = ndi.label(mask)
        markers = morphology.dilation(
                markers,
                selem=morphology.ball(seed_dilation_radius)
        )
        labels = watershed(mem_pred_data, markers)

    return Labels(labels, name='cell_labels')


viewer.window.add_dock_widget(clear_layers, area='right')
viewer.window.add_dock_widget(open_next_image, area='right')
viewer.window.add_dock_widget(generate_seeds_from_nuclei, area='right')
viewer.window.add_dock_widget(run_seeded_watershed, area='right')
viewer.window.add_dock_widget(save_layer, area='right')

napari.run()
