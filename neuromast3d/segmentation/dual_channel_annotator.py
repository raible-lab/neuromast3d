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
import yaml


def execute_step(config):
    raw_dir = Path(config['membrane_segmentation']['raw_dir'])
    nuc_labels_dir = Path(config['membrane_segmentation']['nuc_labels_dir'])
    mem_pred_dir = Path(config['membrane_segmentation']['mem_pred_dir'])
    output_dir = Path(config['membrane_segmentation']['output_dir'])

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
    logger.info('raw dir is %s', config['raw_dir'])
    logger.info('config settings are %s', config['membrane_segmentation']

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
        open_existing_labels: bool = False,
        existing_suffix: str = None
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

        if open_existing_labels:
            try:
                reader = AICSImage(f'{output_dir}/{img_id}_{existing_suffix}.tiff')
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


    @magicgui(
            call_button='Save result to output dir',
            img_id={'choices': list_of_img_ids}
    )
    def save_layers_merged(
            nuc_labels_layer: Labels,
            cell_labels_layer: Labels,
            img_id: Union[str, None],
            suffix: str,
            overwrite: bool
    ):
        '''Save selected layers to output directory as multichannel image.'''

        if img_id is not None:
            # Merge the two labels into a multichannel image
            nuc_labels_data = np.expand_dims(nuc_labels_layer.data, axis=0)
            cell_labels_data = np.expand_dims(cell_labels_layer.data, axis=0)
            merged_labels = np.concatenate((nuc_labels_data, cell_labels_data), axis=0)

            # Save to output dir and log this id as finished
            save_path = output_dir / f'{img_id}_{suffix}.tiff'
            writer = ome_tiff_writer.OmeTiffWriter(save_path, overwrite_file=overwrite)
            writer.save(merged_labels, dimension_order='CZYX')
            logger.info(
                    '%s final results saved at %s',
                    img_id,
                    save_path
            )


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


    @magicgui(call_button='Remove small objects')
    def remove_small_objects_wrapper(
            image: Labels,
            min_size: int = 64,
            connectivity: int = 1,
            make_xor_mask: bool = False
            ) -> List[napari.types.LayerDataTuple]:
        '''Remove small objects from label image'''
        if viewer.layers:
            # idk if this if statement should be more specific? TODO
            # This whole script could really use some cleanup tho...
            cleaned = morphology.remove_small_objects(
                    image.data,
                    min_size=min_size,
                    connectivity=connectivity
            )

            if make_xor_mask:
                # To more easily see what was removed
                image_as_binary = np.where(image.data > 0, 1, 0)
                cleaned_as_binary = np.where(cleaned > 0, 1, 0)
                xor_mask = np.logical_xor(image_as_binary, cleaned_as_binary)

                return [(cleaned, {'name': f'{image.name}_cleaned'}, 'labels'),
                        (xor_mask, {'name': 'cleaned_xor_mask'}, 'image')]

            else:
                return [(cleaned, {'name': f'{image.name}_cleaned'}, 'labels')]


    @magicgui(call_button='Create watershed lines')
    def create_watershed_lines(
            labels: Labels,
            connectivity: int = 2,
            mode: str = 'thick',
            blur: bool = False,
    ) -> Image:
        '''Find boundaries of cell labels'''
        if viewer.layers['cell_labels']:
            watershed_lines = find_boundaries(
                    labels.data,
                    connectivity=connectivity,
                    mode=mode
                    )

            if blur:
                # To use similar method from PlantSeg
                watershed_lines = filters.gaussian(
                        watershed_lines,
                        sigma=1,
                        preserve_range=True
                        )
                watershed_lines[watershed_lines >= 0.5] = 1
                watershed_lines[watershed_lines < 0.5] = 0

        return Image(watershed_lines)


    def update_img_id(event):
        save_layer.img_id.value = event.value
        save_layers_merged.img_id.value = event.value


    viewer.window.add_dock_widget(clear_layers, area='right')
    viewer.window.add_dock_widget(open_next_image, area='right')
    viewer.window.add_dock_widget(generate_seeds_from_nuclei, area='right')
    viewer.window.add_dock_widget(run_seeded_watershed, area='right')
    viewer.window.add_dock_widget(remove_small_objects_wrapper, area='right')
    viewer.window.add_dock_widget(create_watershed_lines, area='right')
    viewer.window.add_dock_widget(save_layer, area='left')
    viewer.window.add_dock_widget(save_layers_merged, area='left')

    open_next_image.img_id.changed.connect(update_img_id)

    napari.run()


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(
            description='Interactive membrane/nucleus segmentation'
    )
    parser.add_argument('config', help='path to config file for segmentation')
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.Loader)
    execute_step(config)


if __name__ == '__main__':
    main()
