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
from magicgui import magicgui
import napari
from napari.layers import Image, Labels, Layer, Shapes
from napari.types import ImageData
import numpy as np
from skimage import morphology, filters
import yaml

from neuromast3d.segmentation.utils import dt_watershed
from neuromast3d.step_utils import check_dir_exists


def execute_step(config):
    raw_dir = Path(config['raw_dir'])
    mask_dir = Path(config['nucleus_segmentation']['nuc_pred_dir'])
    nuc_threshold = config['nucleus_segmentation']['nuc_pred_threshold']
    sigma = config['nucleus_segmentation']['sigma']
    min_distance = config['nucleus_segmentation']['min_distance']

    if config['nucleus_segmentation']['split_nuclei']:
        boundary_dir = Path(config['nucleus_segmentation']['split_nuclei']['boundary_dir'])
        mem_threshold = config['nucleus_segmentation']['split_nuclei']['mem_pred_threshold']
        mode = config['nucleus_segmentation']['split_nuclei']['mode']

    output_dir = Path(config['nucleus_segmentation']['output_dir'])

    # Check raw and mask directories exist
    for directory in [raw_dir, mask_dir]:
        check_dir_exists(directory)

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

    # Initialize viewer
    viewer = napari.Viewer()

    @magicgui(call_button='Clear current viewer')
    def clear_layers():
        if viewer.layers:
            viewer.layers.clear()


    @magicgui(call_button='Read raw img', img_id={'choices': list_of_img_ids})
    def open_raw_image(
            raw_dir: Path = raw_dir,
            img_id=list_of_img_ids[0]
    ) -> Image:
        path = Path(f'{raw_dir}/{img_id}.tiff')
        reader = AICSImage(path)
        raw_img = reader.get_image_data('CZYX', S=0, T=0)
        return Image(raw_img)


    @magicgui(call_button='Read seg img', img_id={'choices': list_of_img_ids})
    def open_seg_image(
            seg_dir: Path = mask_dir,
            img_id=list_of_img_ids[0]
    ) -> Image:
        path = seg_dir / f'{img_id}_struct_segmentation.tiff'
        reader = AICSImage(path)
        seg_img = reader.get_image_data('ZYX', S=0, T=0, C=0)
        return Image(seg_img)


    @magicgui(call_button='Read boundary img', img_id={'choices': list_of_img_ids})
    def open_boundary_image(
            boundary_dir: Path = boundary_dir,
            img_id=list_of_img_ids[0]
    ) -> Image:
        path = boundary_dir / f'{img_id}_struct_segmentation.tiff'
        reader = AICSImage(path)
        boundary_img = reader.get_image_data('ZYX', S=0, T=0, C=0)
        return Image(boundary_img)


    @magicgui(call_button='Apply threshold')
    def apply_threshold(
            layer: ImageData,
            threshold: float = nuc_threshold
    ) -> Image:
        if layer is not None:
            mask = layer > threshold
            return Image(mask)

    if mode == 'automatic':
        @magicgui(call_button='Split nuclei with membranes')
        def split_nuclei_using_membranes(
                nuc_mask: Image,
                mem_pred: Image,
                mem_threshold: float = mem_threshold,
        ) -> Image:
            # TODO: check if this if statement makes sense
            if nuc_mask and mem_pred is not None:
                mem_binary = mem_pred.data > mem_threshold
                nuclei_split = np.where(mem_binary, 0, nuc_mask.data)
                return Image(nuclei_split)

    elif mode == 'interactive':
        @magicgui(call_button='Split nuclei interactively')
        def split_nuclei_interactively(
                raw_img: ImageData,
                nuc_mask: ImageData,
                mem_mask: ImageData,
                shapes: Shapes
        ) -> Image:
            if shapes is not None:
                rois = shapes.to_labels(raw_img.shape) > 0
                mem_mask_selection = mem_mask*rois
                nuclei_split_manual = np.where(mem_mask_selection, 0, nuc_mask)
            return Image(nuclei_split_manual)


    @magicgui(call_button='Run DT watershed')
    def run_dt_watershed(
            image: ImageData,
            sigma: int = sigma,
            min_distance: int = min_distance
    ) -> Labels:
        if image.ndim == 4:
            image = image[0, :, :, :]
        results = dt_watershed(image, sigma=sigma, min_distance=min_distance)
        return Labels(results)


    @magicgui(call_button='Save layer')
    def save_layer(layer: Layer, img_id: str, filename: Path = output_dir):
        if layer:
            raw_save_path = Path(f'{output_dir}/raw_nuc_labels')
            raw_save_path.mkdir(parents=True, exist_ok=True)
            writer = ome_tiff_writer.OmeTiffWriter(
                    raw_save_path/f'{img_id}_rawlabels.tiff',
                    overwrite_file=True
            )
            writer.save(layer.data, dimension_order='ZYX')
            logger.info('%s raw labels saved at %s', img_id, raw_save_path)


    def update_img_id(event):
        open_seg_image.img_id.value = event.value
        save_layer.img_id.value = event.value
        open_boundary_image.img_id.value = event.value


    viewer.window.add_dock_widget(clear_layers, area='left')
    viewer.window.add_dock_widget(open_raw_image, area='right')
    viewer.window.add_dock_widget(open_seg_image, area='right')
    viewer.window.add_dock_widget(open_boundary_image, area='right')
    viewer.window.add_dock_widget(apply_threshold, area='right')
    viewer.window.add_dock_widget(split_nuclei_interactively, area='right')
    viewer.window.add_dock_widget(run_dt_watershed, area='right')
    viewer.window.add_dock_widget(save_layer, area='left')

    open_raw_image.img_id.changed.connect(update_img_id)

    napari.run()


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(
            description='DT watershed segmentation of nucleus mask'
    )
    parser.add_argument('config', help='path to config file for segmentation')
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=yaml.Loader)
    execute_step(config)


if __name__ == '__main__':
    main()
