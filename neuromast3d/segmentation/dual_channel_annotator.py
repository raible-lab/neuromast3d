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
from napari.layers import Image, Labels, Points, Layer
from napari.types import LabelsData
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, filters
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries, watershed


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
raw_dir = Path(args.raw_dir)
nuc_labels_dir = Path(args.nuc_labels_dir)
mem_pred_dir = Path(args.mem_pred_dir)
output_dir = Path(args.output_dir)

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Save command line arguments into log file and output to console
logger = logging.getLogger(__name__)
log_file_path = output_dir / 'seg.log'
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
list_of_img_ids = [fn.stem for fn in raw_dir.glob('*.tiff')]

# Initialize napari viewer
viewer = napari.Viewer()


@magicgui(call_button='Clear current viewer')
def clear_layers():
    '''Clear layers in the current viewer.'''
    if viewer.layers:
        viewer.layers.clear()
    return


@magicgui(
        call_button='Open images',
        img_id={'choices': list_of_img_ids},
        raw_filename={'label': 'Pick a raw file'},
        nuc_seg_filename={'label': 'Pick a nuc seg file'},
        mem_pred_filename={'label': 'Pick a mem seg file'}
)
def open_next_image(
    img_id=list_of_img_ids[0],
    raw_filename: Path = raw_dir,
    nuc_seg_filename: Path = nuc_labels_dir,
    mem_pred_filename: Path = mem_pred_dir,
    use_temp: bool = False
) -> List[napari.types.LayerDataTuple]:

    # Open raw image
    reader = AICSImage(raw_filename)
    image = reader.get_image_data('CZYX', S=0, T=0)

    # Open nuclear labels
    reader = AICSImage(nuc_seg_filename)
    nuc_labels = reader.get_image_data('ZYX', C=0, S=0, T=0)

    # Open membrane boundary predictions
    reader = AICSImage(mem_pred_filename)
    mem_predictions = reader.get_image_data('ZYX', C=0, S=0, T=0)

    # Open cell labels (if existing)
    cell_labels = np.zeros_like(nuc_labels)

    if use_temp:
        try:
            reader = AICSImage(f'{output_dir}/{img_id}_temp.tiff')
            nuc_labels = reader.get_image_data('ZYX', C=0, S=0, T=0)
            cell_labels = reader.get_image_data('ZYX', C=1, S=0, T=0)
        except FileNotFoundError:
            print('Temp file not found, falling back to original')

    return [(image, {'name': 'raw', 'blending': 'additive'}, 'image'),
            (nuc_labels, {'name': 'nuc_labels'}, 'labels'),
            (mem_predictions, {'name': 'mem_predictions', 'blending': 'additive'}, 'image'),
            (cell_labels, {'name': 'cell_labels'}, 'labels')]


@magicgui(
        call_button='Save single layer',
        layer={'choices': viewer.layers},
        img_id={'choices': list_of_img_ids},
        filename={'label': 'Save in'}
)
def save_layer(layer: Layer, img_id: str, filename: Path = output_dir):
    if layer:
        suffix = layer.name
        save_path = filename / f'{img_id}_{suffix}.tiff'
        writer = ome_tiff_writer.OmeTiffWriter(save_path)
        writer.save(layer.data, dimension_order='ZYX')
        logger.info(
                '%s %s saved at %s',
                img_id,
                layer.name,
                save_path
        )

# TODO: add 'suffix' option to allow saving multiple files
# And update open next image with dialog boxes allowing to choose files
@magicgui(
        call_button='Save result to output dir',
        img_id={'choices': list_of_img_ids}
)
def save_layers_merged(
        nuc_labels_layer: Labels,
        cell_labels_layer: Labels,
        img_id: Union[str, None],
        finished: bool = False
):
    '''Save selected layers to output directory as multichannel image.'''

    if img_id is not None:
        # Merge the two labels into a multichannel image
        nuc_labels_data = np.expand_dims(nuc_labels_layer.data, axis=0)
        cell_labels_data = np.expand_dims(cell_labels_layer.data, axis=0)
        merged_labels = np.concatenate((nuc_labels_data, cell_labels_data), axis=0)

        if finished:
            # Save to output dir and log this id as finished
            save_path = output_dir / f'{img_id}_finished.tiff'
            writer = ome_tiff_writer.OmeTiffWriter(save_path)
            writer.save(merged_labels, dimension_order='CZYX')
            logger.info(
                    '%s final results saved at %s',
                    img_id,
                    save_path
            )

        else:
            # Save temporary results, overwriting any already in the output_dir
            save_path = output_dir / f'{img_id}_temp.tiff'
            writer = ome_tiff_writer.OmeTiffWriter(save_path, overwrite_file=True)
            writer.save(merged_labels, dimension_order='CZYX')
            logger.info(
                    '%s temp results saved at %s',
                    img_id,
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
        seed_dilation_radius: int = 10,
        connectivity: int = 2
) -> napari.types.LayerDataTuple:
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
        labels = watershed(mem_pred_data, markers, connectivity=connectivity)

    return (labels, {'name': 'cell_labels'}, 'labels')


@magicgui(call_button='Create watershed lines')
def create_watershed_lines(labels: Labels) -> Image:
    '''Find boundaries of cell labels'''
    if viewer.layers['cell_labels']:
        watershed_lines = find_boundaries(labels.data)

    return Image(watershed_lines)


def update_img_id(event):
    save_layer.img_id.value = event.value
    # save_layer.img_id.value = event.value.img_id


viewer.window.add_dock_widget(clear_layers, area='right')
viewer.window.add_dock_widget(open_next_image, area='right')
viewer.window.add_dock_widget(generate_seeds_from_nuclei, area='right')
viewer.window.add_dock_widget(run_seeded_watershed, area='right')
viewer.window.add_dock_widget(create_watershed_lines, area='right')
viewer.window.add_dock_widget(save_layer, area='right')

open_next_image.img_id.changed.connect(update_img_id)

napari.run()
