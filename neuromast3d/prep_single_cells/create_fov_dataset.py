#!usr/bin/env python3
# -*- coding: utf-8 -*-

""" Prepare a fov dataset (without splitting into single cells

Allows addition of manual values to the FOV dataset to use for tilt correction
etc.
"""

import argparse
from pathlib import Path
import re
import sys

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
from aicsimageprocessing import resize, resize_to
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from skimage.morphology import ball, binary_closing, remove_small_objects
import yaml

from neuromast3d.alignment.utils import find_major_axis_by_pca
from neuromast3d.prep_single_cells.utils import apply_3d_rotation, rotate_image_3d
from neuromast3d.step_utils import read_raw_and_seg_img, step_logger, check_dir_exists


def create_name_dict(raw_channel_ids, seg_channel_ids):
    name_dict = {
            'crop_raw': [*raw_channel_ids.keys()],
            'crop_seg': [*seg_channel_ids.keys()]
    }
    return name_dict


def create_fov_dataframe(raw_files, seg_files, og_files, raw_channel_ids, seg_channel_ids):
    channel_ids = {**raw_channel_ids, **seg_channel_ids}
    name_dict = create_name_dict(raw_channel_ids, seg_channel_ids)
    fov_info = []
    for fn in raw_files:
        raw_img_name = fn.stem
        pattern = re.compile(raw_img_name)
        seg_img_path = [fn for fn in seg_files if pattern.match(fn.stem)]
        og_img_path = [fn for fn in og_files if pattern.match(fn.stem)]

        try:
            reader = AICSImage(og_img_path[0])
            pixel_size = reader.get_physical_pixel_size()
        except FileNotFoundError:
            pass

        fov_info.append(
                {
                    'NM_ID': raw_img_name,
                    'SourceReadPath': fn,
                    'SegmentationReadPath': seg_img_path[0],
                    'pixel_size_xyz': pixel_size,
                    'name_dict': name_dict,
                    **channel_ids
                }
        )

    fov_dataset = pd.DataFrame(fov_info)
    return fov_dataset


def create_whole_nm_mask(seg_img, channel):
    seg_img = remove_small_objects(seg_img, min_size=200)
    whole_nm_mask = seg_img[channel, :, :, :] > 0
    whole_nm_mask = whole_nm_mask.astype(np.uint8)
    whole_nm_mask = whole_nm_mask*255
    return whole_nm_mask


def apply_autorotation(fov_dataset, output_dir):
    # Iterate through all the files
    rot_angles = []
    for fov in fov_dataset.itertuples(index=False):

        raw_img, seg_img = read_raw_and_seg_img(
                fov.SourceReadPath,
                fov.SegmentationReadPath
        )
        whole_nm_mask = create_whole_nm_mask(seg_img, fov.cell_seg)
        pixel_size_x, pixel_size_y, _ = fov.pixel_size_xyz
        assert pixel_size_x == pixel_size_y

        # Calculate 3d rotation angles for tilt correction
        yaw, pitch, roll = rotate_image_3d(whole_nm_mask)

        # Apply 3d rotation to raw and seg images
        raw_img_rot = apply_3d_rotation(raw_img, yaw, pitch, roll)
        seg_img_rot = apply_3d_rotation(seg_img, yaw, pitch, roll)

        rot_angles.append(
                {
                    'NM_ID': fov.NM_ID,
                    'angle_1': yaw,
                    'angle_2': pitch,
                    'angle_3': roll
                }
        )

        raw_rot_path = output_dir / f'{fov.NM_ID}_raw_rot.tiff'
        writer = ome_tiff_writer.OmeTiffWriter(raw_rot_path)
        writer.save(raw_img_rot, dimension_order='CZYX')

        seg_rot_path = output_dir / f'{fov.NM_ID}_seg_rot.tiff'
        writer = ome_tiff_writer.OmeTiffWriter(seg_rot_path)
        writer.save(seg_img_rot, dimension_order='CZYX')
        return rot_angles


def get_channel_ids(config):
    raw_channel_ids = config['raw_channels']
    seg_channel_ids = config['seg_channels']
    return raw_channel_ids, seg_channel_ids


def execute_step(config):
    # This function can be called as part of running a workflow
    # or as a standalone script (e.g. if using main() function)
    step_name = 'create_fov_dataset'

    original_dir = Path(config['create_fov_dataset']['original_dir'])
    raw_dir = Path(config['raw_dir'])
    seg_dir = Path(config['create_fov_dataset']['seg_dir'])
    output_dir = Path(config['project_dir'])
    rotate_auto = config['create_fov_dataset']['autorotate']
    raw_channel_ids, seg_channel_ids = get_channel_ids(config['channels'])

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save command line arguments into logfile
    logger = step_logger(step_name, output_dir)
    logger.info(sys.argv)

    # Check paths exist
    for dir_name in raw_dir, seg_dir:
        check_dir_exists(dir_name)

    # Find all files in source directories and gather image filenames into list
    raw_files = list(raw_dir.glob('*.tiff'))
    seg_files = list(seg_dir.glob('*.tiff'))
    og_files = list(original_dir.glob('*.czi'))

    if not len(raw_files) == len(seg_files):
        print('Number of raw files does not match number of seg files.')
        sys.exit()

    # Create initial fov dataframe (where every row is a neuromast)
    fov_dataset = create_fov_dataframe(raw_files, seg_files, og_files, raw_channel_ids, seg_channel_ids)
    fov_dataset.to_csv(output_dir / 'fov_dataset.csv')

    if rotate_auto:

        rot_angles = apply_autorotation(fov_dataset, output_dir)
        rot_angle_df = pd.DataFrame(rot_angles)
        rot_angle_df = fov_dataset.merge(rot_angle_df, on='NM_ID')
        path_to_angle_df = output_dir / 'fov_dataset_with_rot.csv'
        rot_angle_df.to_csv(path_to_angle_df)


def main():
    parser = argparse.ArgumentParser(
            description='Create fov dataset script'
    )
    parser.add_argument('config', help='path to config file')
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.Loader)
    execute_step(config)


if __name__ == '__main__':
    main()
