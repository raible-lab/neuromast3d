#!usr/bin/env python3
# -*- coding: utf-8 -*-

""" Prepare single cells for cvapipe_analysis (dual channel version) """

import argparse
from ast import literal_eval
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
from aicsimageprocessing import resize, resize_to

from neuromast3d.prep_single_cells.utils import apply_3d_rotation
from neuromast3d.prep_single_cells.create_fov_dataset import step_logger

def inherit_labels(dna_mask_bw, mem_seg):
    dna_mask_label = np.zeros_like(mem_seg)
    dna_mask_label[dna_mask_bw > 0] = 1
    dna_mask_label = dna_mask_label * mem_seg
    nuc_seg = dna_mask_label
    return nuc_seg


def remove_small_labels(mem_seg_whole, nuc_seg_whole):
    cell_label_list = list(np.unique(mem_seg_whole[mem_seg_whole > 0]))
    cell_label_list_copy = cell_label_list.copy()
    for count, label in enumerate(cell_label_list):
        single_cell_mem = nuc_seg_whole == label
        single_cell_nuc = mem_seg_whole == label

        # These numbers are guessed as reasonable thresholds
        if(
                np.count_nonzero(single_cell_mem) < 500
                or np.count_nonzero(single_cell_nuc) < 100
        ):
            cell_label_list_copy.remove(label)

    # A bit awkward, but we can't remove items from a list we're iterating over
    cell_label_list = cell_label_list_copy.copy()
    return cell_label_list


def create_cropping_roi(mem_seg):
    z_range = np.where(np.any(mem_seg, axis=(1, 2)))
    y_range = np.where(np.any(mem_seg, axis=(0, 2)))
    x_range = np.where(np.any(mem_seg, axis=(0, 1)))
    z_range = z_range[0]
    y_range = y_range[0]
    x_range = x_range[0]
    roi = [
            max(z_range[0] - 10, 0),
            min(z_range[-1] + 12, mem_seg.shape[0]),  # not sure why 12
            max(y_range[0] - 40, 0),
            min(y_range[-1] + 40, mem_seg.shape[1]),
            max(x_range[0] - 40, 0),
            min(x_range[-1] + 40, mem_seg.shape[2])
    ]
    return roi


def crop_to_roi(image, roi):
    image = image[roi[0]: roi[1], roi[2]: roi[3], roi[4]: roi[5]]
    return image


def standardize_channel_order(row):
    raw_structures = ['dna', 'membrane']
    seg_structures = ['dna_seg', 'cell_seg']
    raw_structures_reordered = [raw_structures[i] for i in [row.RawNucChannelIndex, row.RawMemChannelIndex]]
    seg_structures_reordered = [seg_structures[i] for i in [row.SegNucChannelIndex, row.SegMemChannelIndex]]
    return raw_structures_reordered, seg_structures_reordered


def read_fov_images(row):
    reader_raw = AICSImage(row.SourceReadPath)
    mem_raw = reader_raw.get_image_data('ZYX', S=0, T=0, C=row.RawMemChannelIndex)
    nuc_raw = reader_raw.get_image_data('ZYX', S=0, T=0, C=row.RawNucChannelIndex)

    reader_seg = AICSImage(row.SegmentationReadPath)
    mem_seg = reader_seg.get_image_data('ZYX', S=0, T=0, C=row.SegMemChannelIndex)
    nuc_seg = reader_seg.get_image_data('ZYX', S=0, T=0, C=row.SegNucChannelIndex)
    return mem_raw, nuc_raw, mem_seg, nuc_seg


def create_single_cell_dataset(fov_dataset, output_dir, rotate_auto=False):
    # Create dir for single cells to go into
    single_cell_dir = output_dir / 'single_cell_masks'
    single_cell_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through cells in each FOV
    cell_meta = []
    for row in fov_dataset.itertuples(index=False):

        # Make dir for this FOV
        current_fov_dir = single_cell_dir / f'{row.NM_ID}'
        current_fov_dir.mkdir(parents=True, exist_ok=True)

        # Get the raw and segmented FOV images
        mem_raw, nuc_raw, mem_seg, nuc_seg = read_fov_images(row)

        # Discard nuclei not included in the membrane
        dna_mask_bw = nuc_seg
        dna_mask_bw[mem_seg == 0] = 0

        # Make nuclei labels match the cell's
        nuc_seg = inherit_labels(dna_mask_bw, mem_seg)

        # Take z_res and xy_res from fov_dataframe
        xy_res, _, z_res = literal_eval(row.pixel_size_xyz)

        # Rescale raw images to isotropic dimenstions
        raw_nuc_whole = resize(
                nuc_raw,
                (z_res / xy_res, xy_res / xy_res, xy_res / xy_res),
                method='bilinear'
        ).astype(np.uint16)

        raw_mem_whole = resize(
                mem_raw,
                (z_res / xy_res, xy_res / xy_res, xy_res / xy_res),
                method='bilinear'
        ).astype(np.uint16)

        # Resize seg images to match
        mem_seg_whole = resize_to(mem_seg, raw_mem_whole.shape, method='nearest')
        nuc_seg_whole = resize_to(nuc_seg, raw_nuc_whole.shape, method='nearest')

        # Remove very small cells from the list
        cell_label_list = remove_small_labels(mem_seg_whole, nuc_seg_whole)

        # Crop and prep the cells
        for label in cell_label_list:
            label_dir = current_fov_dir / f'{label}'
            label_dir.mkdir(parents=True, exist_ok=True)
            mem_seg = mem_seg_whole == label
            nuc_seg = nuc_seg_whole == label

            # Make the cropping roi
            roi = create_cropping_roi(mem_seg)

            # Crop both segmentation channels to membrane roi
            mem_seg = mem_seg.astype(np.uint8)
            mem_seg = crop_to_roi(mem_seg, roi)
            mem_seg[mem_seg > 0] = 255

            nuc_seg = nuc_seg.astype(np.uint8)
            nuc_seg = crop_to_roi(nuc_seg, roi)
            nuc_seg[nuc_seg > 0] = 255

            # Merge channels and save
            seg_merged = np.stack([nuc_seg, mem_seg], axis=0)

            # Apply tilt correction if desired
            # TODO: should this be done here?
            # Or should it be done during another step, i.e. alignment?
            if rotate_auto:
                seg_merged = apply_3d_rotation(
                        seg_merged,
                        row.angle_1,
                        row.angle_2,
                        row.angle_3
                )

            crop_seg_path = label_dir / 'segmentation.ome.tif'
            writer = ome_tiff_writer.OmeTiffWriter(crop_seg_path)
            writer.save(seg_merged, dimension_order='CZYX')

            # Crop both channels of the raw image
            raw_nuc = crop_to_roi(raw_nuc_whole, roi)
            raw_mem = crop_to_roi(raw_mem_whole, roi)

            # Merge channels and save
            raw_merged = np.stack([raw_nuc, raw_mem], axis=0)

            # Apply tilt correction if desired
            # Similar TODO as above applies
            if rotate_auto:
                raw_merged = apply_3d_rotation(
                        raw_merged,
                        row.angle_1,
                        row.angle_2,
                        row.angle_3
                )

            crop_raw_path = label_dir / 'raw.ome.tif'
            writer = ome_tiff_writer.OmeTiffWriter(crop_raw_path)
            writer.save(raw_merged, dimension_order='CZYX')
            cell_id = f'{row.NM_ID}_{label}'

            # Add name dict - may need to alter for 3 channel images
            raw_structures, seg_structures = standardize_channel_order(row)
            name_dict = {
                     'crop_raw': raw_structures,
                     'crop_seg': seg_structures,
             }

            # If no structure name, add NoStr as a placeholder for future steps
            # Note: avoid using NA or other pandas default_na_values
            if 'Gene' in fov_dataset:
                structure_name = row.Gene
            else:
                structure_name = 'NoStr'
            cell_meta.append(
                    {
                         'CellId': cell_id,
                         'label': label,
                         'roi': roi,
                         'crop_raw': crop_raw_path,
                         'crop_seg': crop_seg_path,
                         'pixel_size_xyz': row.pixel_size_xyz,
                         'scale_micron': [xy_res, xy_res, xy_res],
                         'fov_id': row.NM_ID,
                         'fov_path': row.SourceReadPath,
                         'fov_seg_path': row.SegmentationReadPath,
                         'name_dict': name_dict,
                         'structure_name': structure_name,
                         'RawNucChannelIndex': row.RawNucChannelIndex,
                         'RawMemChannelIndex': row.RawMemChannelIndex,
                         'SegNucChannelIndex': row.SegNucChannelIndex,
                         'SegMemChannelIndex': row.SegMemChannelIndex
                    }
            )
    return cell_meta


def execute_step(config):
    step_name = 'prep_single_cells'
    project_dir = Path(config['create_fov_dataset']['output_dir'])
    output_dir = project_dir / 'prep_single_cells'
    rotate_auto = config['create_fov_dataset']['autorotate']
    if rotate_auto:
        path_to_fov_dataset = Path(config['create_fov_dataset']['output_dir']) / 'fov_dataset_with_rot.csv'

    else:
        path_to_fov_dataset = Path(config['create_fov_dataset']['output_dir']) / 'fov_dataset.csv'

    # Create destination directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save command line arguments into logfile
    logger = step_logger(step_name, output_dir)
    logger.info(sys.argv)

    # Check path to fov dataset exists
    assert path_to_fov_dataset.exists

    # Read fov dataset
    fov_dataset = pd.read_csv(path_to_fov_dataset)

    # Create the cell dataset
    cell_meta = create_single_cell_dataset(fov_dataset, output_dir)

    # Save cell dataset (every row is a cell)
    df_cell_meta = pd.DataFrame(cell_meta)
    path_to_manifest = output_dir / 'cell_manifest.csv'
    df_cell_meta.to_csv(path_to_manifest)


def main():
    # Command line arguments
    parser = argparse.ArgumentParser(
            description='script to prep single cells'
    )
    parser.add_argument('config', help='path to config file')
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.Loader)
    execute_step(config)


if __name__ == '__main__':
    main()
