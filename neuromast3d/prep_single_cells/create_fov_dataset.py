#!usr/bin/env python3
# -*- coding: utf-8 -*-

""" Prepare a fov dataset (without splitting into single cells

Allows addition of manual values to the FOV dataset to use for tilt correction
etc.
"""

import argparse
import logging
from pathlib import Path
import re
import sys
from typing import Tuple

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
from aicsimageprocessing import resize, resize_to
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
from skimage.morphology import ball, binary_closing, remove_small_objects
import yaml

from neuromast3d.alignment.utils import find_major_axis_by_pca, prepare_vector_for_napari
from utils import apply_3d_rotation, rotate_image_3d


'''
# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('original_dir', help='directory containing OG images')
parser.add_argument('raw_dir', help='directory containing raw images')
parser.add_argument('seg_dir', help='directory containing segmented images')
parser.add_argument('output_dir', help='directory in which to save outputs')
parser.add_argument('raw_nuc_ch_index', type=int)
parser.add_argument('raw_mem_ch_index', type=int)
parser.add_argument('seg_nuc_ch_index', type=int)
parser.add_argument('seg_mem_ch_index', type=int)
parser.add_argument(
        '-r',
        '--rotate_auto',
        help='apply automatic rotation based on major axes found by PCA',
        action='store_true'
)

# Parse arguments
args = parser.parse_args()
original_dir = Path(args.original_dir)
raw_dir = Path(args.raw_dir)
seg_dir = Path(args.seg_dir)
output_dir = Path(args.output_dir)
'''


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
            description='Create fov dataset script'
    )
    parser.add_argument('config', help='path to config file')
    args = parser.parse_args()

    # Read config
    config = yaml.load(open(args.config), Loader=yaml.Loader)
    original_dir = Path(config['create_fov_dataset']['original_dir'])
    raw_dir = Path(config['segmentation']['raw_dir'])
    seg_dir = Path(config['create_fov_dataset']['seg_dir'])
    output_dir = Path(config['create_fov_dataset']['output_dir'])
    raw_nuc_ch_index = config['segmentation']['raw_nuc_ch']
    raw_mem_ch_index = config['segmentation']['raw_mem_ch']
    seg_nuc_ch_index = config['create_fov_dataset']['seg_nuc_ch']
    seg_mem_ch_index = config['create_fov_dataset']['seg_mem_ch']
    rotate_auto = config['create_fov_dataset']['autorotate']

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save command line arguments into logfile
    log_file_path = output_dir / 'create_fov_dataset.log'
    logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format='%(asctime)s %(message)s'
    )
    logger.info(sys.argv)

    # Check paths exist
    if not raw_dir.is_dir():
        print('Raw dir does not exist')
        sys.exit()

    if not seg_dir.is_dir():
        print('Seg dir does not exist')
        sys.exit()

    # Find all files in source directories and gather image filenames into list
    raw_files = list(raw_dir.glob('*.tiff'))
    seg_files = list(seg_dir.glob('*.tiff'))
    og_files = list(original_dir.glob('*.czi'))

    if not len(raw_files) == len(seg_files):
        print('Number of raw files does not match number of seg files.')
        sys.exit()

    # Create initial fov dataframe (where every row is a neuromast)

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
                    'RawNucChannelIndex': raw_nuc_ch_index,
                    'RawMemChannelIndex': raw_mem_ch_index,
                    'SegNucChannelIndex': seg_nuc_ch_index,
                    'SegMemChannelIndex': seg_mem_ch_index
                }
        )

    fov_dataset = pd.DataFrame(fov_info)
    fov_dataset.to_csv(output_dir / 'fov_dataset.csv')

    if rotate_auto:

        # Iterate through all the files
        rot_angles = []
        for fov in fov_dataset.itertuples(index=False):

            # Read in raw and seg images
            reader = AICSImage(fov.SourceReadPath)
            raw_img = reader.get_image_data('CZYX', S=0, T=0)

            reader = AICSImage(fov.SegmentationReadPath)
            seg_img = reader.get_image_data('CZYX', S=0, T=0)

            # Clean up small artifacts
            seg_img = remove_small_objects(seg_img, min_size=200)

            # Merge cell labels together to create whole neuromast mask
            pixel_size_x, pixel_size_y, pixel_size_z = fov.pixel_size_xyz
            assert pixel_size_x == pixel_size_y
            whole_nm_mask = seg_img[fov.SegMemChannelIndex, :, :, :] > 0

            """

            whole_nm_mask = resize(
                    whole_nm_mask,
                    (
                        pixel_size_z / pixel_size_x,
                        pixel_size_y / pixel_size_x,
                        pixel_size_x / pixel_size_x
                    ),
                    method='bilinear'
            )

            """

            # Clean up the mask a bit
            # whole_nm_mask = binary_closing(whole_nm_mask, ball(5))
            whole_nm_mask = whole_nm_mask.astype(np.uint8)
            whole_nm_mask = whole_nm_mask*255

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

        rot_angle_df = pd.DataFrame(rot_angles)
        rot_angle_df = fov_dataset.merge(rot_angle_df, on='NM_ID')
        path_to_angle_df = output_dir / 'fov_dataset_with_rot.csv'
        rot_angle_df.to_csv(path_to_angle_df)



"""
# debugging stuff
nm_centroid = center_of_mass(whole_nm_mask)
vec1 = prepare_vector_for_napari(np.flip(eigenvecs[0]), origin=nm_centroid, scale=100)
vec2 = prepare_vector_for_napari(np.flip(eigenvecs[1]), origin=nm_centroid, scale=100)
vec3 = prepare_vector_for_napari(np.flip(eigenvecs[2]), origin=nm_centroid, scale=100)

#eigenvecs_final = find_major_axis_by_pca(nm_mask_rot_1, threed=True)
#eigenvecs_final = find_major_axis_by_pca(nm_mask_rot_2, threed=True)
eigenvecs_final = find_major_axis_by_pca(nm_rot_final, threed=True)
vec11 = prepare_vector_for_napari(np.flip(eigenvecs_final[0]), origin=nm_centroid, scale=100)
vec22 = prepare_vector_for_napari(np.flip(eigenvecs_final[1]), origin=nm_centroid, scale=100)
vec33 = prepare_vector_for_napari(np.flip(eigenvecs_final[2]), origin=nm_centroid, scale=100)

viewer.add_vectors(vec1)
viewer.add_vectors(vec2)
viewer.add_vectors(vec3)
viewer.add_vectors(vec11)
viewer.add_vectors(vec22)
viewer.add_vectors(vec33)
viewer.add_points(nm_centroid)

napari.run()
"""
