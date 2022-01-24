#!usr/bin/env python3
# -*- coding: utf-8 -*-

""" Prepare single cells for cvapipe_analysis (dual channel version) """

import argparse
from ast import literal_eval
from functools import partial
from pathlib import Path
import sys

from aicsimageio.writers import ome_tiff_writer
from aicsimageprocessing import resize, resize_to
import numpy as np
import pandas as pd
import yaml

from neuromast3d.prep_single_cells.utils import apply_3d_rotation
from neuromast3d.prep_single_cells.create_fov_dataset import step_logger, read_raw_and_seg_img


def inherit_labels(dna_mask_bw, mem_seg):
    dna_mask_label = np.zeros_like(mem_seg)
    dna_mask_label[dna_mask_bw > 0] = 1
    dna_mask_label = dna_mask_label * mem_seg
    nuc_seg = dna_mask_label
    return nuc_seg


def remove_small_labels(label_img, size_threshold):
    label_list = list(np.unique(label_img[label_img > 0]))
    label_list_copy = label_list.copy()
    for label in label_list:
        single_label = label_img == label
        if np.count_nonzero(single_label) < size_threshold:
            label_list_copy.remove(label)
    return label_list_copy


def remove_small_labels_2ch(mem_seg_whole, nuc_seg_whole):
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
    assert mem_seg.ndim == 3
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


def discard_labels_outside_mask(labels, mask_labels):
    labels_new = labels
    labels_new[mask_labels == 0] = 0
    return labels_new


def apply_function_to_all_channels(image, function):
    # Expects 4 channel image in CZYX dim order
    img_processed = []
    for channel in image:
        ch_processed = function(channel)
        img_processed.append(ch_processed)
    img_processed = np.array(img_processed)
    return img_processed


def create_single_cell_dataset(fov_dataset, output_dir, rotate_auto=False):
    # Create dir for single cells to go into
    single_cell_dir = output_dir / 'single_cell_masks'
    single_cell_dir.mkdir(parents=True, exist_ok=True)

    cell_meta = []
    for row in fov_dataset.itertuples(index=False):

        # Make dir for this FOV
        current_fov_dir = single_cell_dir / f'{row.NM_ID}'
        current_fov_dir.mkdir(parents=True, exist_ok=True)

        raw_img, seg_img = read_raw_and_seg_img(row.SourceReadPath, row.SegmentationReadPath)
        mem_raw = raw_img[row.membrane, :, :, :]
        mem_seg = seg_img[row.cell_seg, :, :, :]
        xy_res, _, z_res = literal_eval(row.pixel_size_xyz)

        # Rescale images to isotropic dimenstions
        raw_mem_whole = resize(
                mem_raw,
                (z_res / xy_res, xy_res / xy_res, xy_res / xy_res),
                method='bilinear'
        ).astype(np.uint16)

        resize_to_mem_raw = partial(resize_to, out_size=raw_mem_whole.shape, method='nearest')
        mem_seg_whole = resize_to_mem_raw(mem_seg)

        raw_mem_whole = np.expand_dims(raw_mem_whole, axis=0)
        mem_seg_whole = np.expand_dims(mem_seg_whole, axis=0)

        if raw_img.shape[0] > 1:
            raw_non_mem = np.delete(raw_img, row.membrane, 0)
            resize_to_isotropic = partial(
                    resize,
                    factor=(z_res / xy_res, xy_res / xy_res, xy_res / xy_res),
                    method='bilinear'
            )
            raw_non_mem = apply_function_to_all_channels(raw_non_mem, resize_to_isotropic).astype(np.uint16)
            raw_whole = np.concatenate([raw_mem_whole, raw_non_mem], axis=0)

        else:
            raw_whole = raw_mem_whole

        if seg_img.shape[0] > 1:
            seg_non_mem = np.delete(seg_img, row.cell_seg, 0)

            discard_labels_outside_cell = partial(discard_labels_outside_mask, mask_labels=mem_seg)
            seg_non_mem = apply_function_to_all_channels(seg_non_mem, discard_labels_outside_cell)

            inherit_labels_from_cell = partial(inherit_labels, mem_seg=mem_seg)
            seg_non_mem = apply_function_to_all_channels(seg_non_mem, inherit_labels_from_cell)

            seg_non_mem = apply_function_to_all_channels(seg_non_mem, resize_to_mem_raw)
            seg_whole = np.concatenate([mem_seg_whole, seg_non_mem], axis=0)

        else:
            seg_whole = mem_seg_whole

        # Remove very small cells from the list
        # TODO: this used to depend on nucleus size too
        # Might want to add that back in?
        cell_label_list = remove_small_labels(mem_seg_whole, size_threshold=500)

        # Crop and prep the cells
        for label in cell_label_list:
            label_dir = current_fov_dir / f'{label}'
            label_dir.mkdir(parents=True, exist_ok=True)
            mem_seg = mem_seg_whole == label
            seg_whole = seg_whole == label
            print('single cell shape is ', mem_seg.shape)

            # Crop all channels to cell_seg roi
            roi = create_cropping_roi(np.squeeze(mem_seg))
            crop_to_mem_seg = partial(crop_to_roi, roi=roi)
            seg_img = seg_whole.astype(np.uint8)
            seg_img = apply_function_to_all_channels(seg_img, crop_to_mem_seg)
            seg_img[seg_img > 0] = 255
            print('cropped single cell shape is ', seg_img.shape)

            # Apply tilt correction if desired
            # TODO: should this be done here?
            # Or should it be done during another step, i.e. alignment?
            if rotate_auto:
                seg_img = apply_3d_rotation(
                        seg_img,
                        row.angle_1,
                        row.angle_2,
                        row.angle_3
                )

            crop_seg_path = label_dir / 'segmentation.ome.tif'
            writer = ome_tiff_writer.OmeTiffWriter(crop_seg_path)
            writer.save(seg_img, dimension_order='CZYX')

            # Crop both channels of the raw image
            raw_img = raw_whole
            raw_img = apply_function_to_all_channels(raw_img, crop_to_mem_seg)

            # Apply tilt correction if desired
            # Similar TODO as above applies
            if rotate_auto:
                raw_img = apply_3d_rotation(
                        raw_img,
                        row.angle_1,
                        row.angle_2,
                        row.angle_3
                )

            crop_raw_path = label_dir / 'raw.ome.tif'
            writer = ome_tiff_writer.OmeTiffWriter(crop_raw_path)
            writer.save(raw_img, dimension_order='CZYX')

            cell_id = f'{row.NM_ID}_{label}'

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
                         'name_dict': row.name_dict,
                         'structure_name': structure_name,
                         'RawNucChannelIndex': row.nucleus,
                         'RawMemChannelIndex': row.membrane,
                         'SegNucChannelIndex': row.nucleus_seg,
                         'SegMemChannelIndex': row.cell_seg
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

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save command line arguments into logfile
    logger = step_logger(step_name, output_dir)
    logger.info(sys.argv)

    assert path_to_fov_dataset.exists
    fov_dataset = pd.read_csv(path_to_fov_dataset)
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
