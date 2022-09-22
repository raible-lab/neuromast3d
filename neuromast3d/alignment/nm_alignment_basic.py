#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Align all cells to same cardinal direction

As suggested by Lorenzo, this script assumes radial symmetry of the neuromast
and does not try to align cells to an organismal axis (e.g. A/P, D/V).
"""

import ast
import argparse
from functools import partial
from locale import normalize
import pathlib
import sys
from typing import Tuple, Union

from aicsimageio import AICSImage
from aicsimageprocessing import resize, resize_to
import numpy as np
import pandas as pd
from skimage.morphology import binary_closing, ball
from sklearn.decomposition import PCA
import scipy.ndimage as ndi
import yaml
from neuromast3d.prep_single_cells.prep_single_cells import apply_function_to_all_channels, inherit_labels
from neuromast3d.prep_single_cells.utils import apply_3d_rotation, rotate_image_3d

from neuromast3d.step_utils import read_raw_and_seg_img, check_dir_exists, step_logger, create_step_dir, save_raw_and_seg_cell
from neuromast3d.alignment.utils import rotate_image_2d_custom


def normalize_centroid(
    image: np.array,
    origin: tuple
):
    if image.ndim != 3:
        raise ValueError(f'Invalid shape of input image {image.shape}. \
                Image must be a single-channel, 3D stack.')

    centroid = ndi.center_of_mass(image)
    centroid_normed = np.subtract(centroid, origin)
    return centroid_normed


def calculate_alignment_angle_2d(
        centroid_normed: tuple,
):
    """Calculate 2d alignment angle for a cell compared to a user-defined
    origin to use as the axis of rotation. The centroid of the image is
    calculated and the origin is subtracted from it to produce a vector
    describing the location of the cell centroid compared to the origin in
    the xy-plane. The rotation angle between this vector and the x-axis
    (3 o'clock position) is then calculated.


    Parameters
    ----------
    image : np.array
        The input image to be rotated. Must be a single channel, 3D stack.
        The centroid of this image will be calculated for the rotation.

    origin : tuple
        The origin around which to perform the rotation. Must be a 3-tuple in
        dimension order z, y, x. Only the x and y coordinates are actually
        used for the rotation.

    make_unique : bool, optional
        Whether to use the arctan2 function for the rotation, which
        preserves information about the original orientation of the cell.
        Default True.

    Returns
        The angle for rotation and the image centroid after subtracting
        the defined origin. The sign of the angle can be positive or negative,
        depending on where the cell centroid was relative to the origin.
        Cells with positive angles should be rotated CW, and those with
        negative angles should be rotated CCW.

    """
    x = centroid_normed[2]
    y = -centroid_normed[1]

    # Calculate angle with atan2 to preserve orientation
    # I think this SHOULD align to the 3 o' clock position?
    angle = 180 * np.arctan2(y, x) / np.pi

    return angle


def calculate_2d_long_axis_angle_to_z_axis(seg_cell, proj_type: str):
    if seg_cell.ndim == 3:
        seg_cell = seg_cell[np.newaxis, :, :, :]
    elif seg_cell.ndim == 4:
        seg_cell = seg_cell
    else:
        raise ValueError('Seg cell ndims must be 3 or 4')

    _, z, y, x = np.nonzero(seg_cell)
    if proj_type == 'xz':
        coords = np.hstack([x.reshape(-1, 1), z.reshape(-1, 1)])
    elif proj_type == 'yz':
        coords = np.hstack([y.reshape(-1, 1), z.reshape(-1, 1)])
    pca = PCA(n_components=2)
    pca = pca.fit(coords)
    eigenvecs = pca.components_
    angle = 180 * np.arctan(eigenvecs[0][0]/eigenvecs[0][1]) / np.pi
    # Angle needs to be negative for downstream steps to work, I think
    # (not entirely sure why, probably a coord system thing)
    angle = -angle
    return angle


def align_cell_xz_long_axis_to_z_axis(raw_cell, seg_cell):
    angle = calculate_2d_long_axis_angle_to_z_axis(seg_cell, 'xz')
    seg_cell_aligned = ndi.rotate(seg_cell, -angle, axes=(1, 3), order=0)
    raw_cell_aligned = ndi.rotate(raw_cell, -angle, axes=(1, 3), order=0)
    return raw_cell_aligned, seg_cell_aligned


def align_cell_yz_long_axis_to_z_axis(raw_cell, seg_cell):
    angle = calculate_2d_long_axis_angle_to_z_axis(seg_cell, 'yz')
    seg_cell_aligned = ndi.rotate(seg_cell, -angle, axes=(1, 2), order=0)
    raw_cell_aligned = ndi.rotate(raw_cell, -angle, axes=(1, 2), order=0)
    return raw_cell_aligned, seg_cell_aligned


def create_fov_dataframe_from_cell_dataframe(cell_df):
    fov_df = cell_df.copy()
    fov_df.drop_duplicates(subset=['fov_id'], keep='first', inplace=True)
    fov_df.drop(
        ['CellId', 'crop_raw_pre_alignment', 'crop_seg_pre_alignment', 'roi'],
        axis=1,
        inplace=True
    )
    return fov_df


def interpolate_fov_in_z(seg_img, pixel_size_xyz):
    nm = seg_img > 0
    xy_res, _, z_res = pixel_size_xyz
    nm = resize(
            seg_img,
            (
                z_res / xy_res,
                xy_res / xy_res,
                xy_res / xy_res
            ),
            method='bilinear'
    )
    seg_img = resize_to(seg_img, nm.shape, method='nearest')
    nm = binary_closing(nm, ball(5))
    nm_centroid = ndi.center_of_mass(nm)
    return seg_img, nm_centroid


def get_alignment_settings(config) -> dict:
    project_dir = pathlib.Path(config['project_dir'])
    settings = {
            'project_dir': project_dir,
            'path_to_manifest': project_dir / 'prep_single_cells/cell_manifest.csv',
            'rot_ch_index': config['alignment']['rot_ch_index'],
            'make_unique': config['alignment']['make_unique'],
            'mode': config['alignment']['mode'],
            'use_channels': config['alignment']['use_channels'],
            'continue_from_previous': config['alignment']['continue_from_previous']
    }
    return settings


def prepare_fov(fov, settings, cell_df):
    print('starting alignment for', fov.fov_id)
    seg_reader = AICSImage(fov.fov_seg_path)

    seg_img = seg_reader.get_image_data('ZYX', S=0, T=0, C=settings['rot_ch_index'])
    pixel_size_xyz = ast.literal_eval(fov.pixel_size_xyz)

    # Inherit labels from cell again (hopefully this doesn't cause memory issues)
    mem_seg = seg_reader.get_image_data('ZYX', S=0, T=0, C=fov.cell_seg)
    seg_img = inherit_labels(seg_img, mem_seg)
    mem_seg = 0

    seg_img, nm_centroid = interpolate_fov_in_z(seg_img, pixel_size_xyz)
    fov_info = {'nm_centroid': nm_centroid, 'fov_id': fov.fov_id}

    current_fov_cells = cell_df[cell_df['fov_id'] == fov.fov_id]

    return current_fov_cells, fov_info, seg_img


def prepare_cell_and_fov_datasets(settings, step_dir):
    cell_df = pd.read_csv(settings['path_to_manifest'], index_col=0)

    # Since we are applying alignment, rename old crop_seg and crop_raw
    # Because we want to use the aligned images in future steps
    cell_df = cell_df.rename(columns={
        'crop_raw': 'crop_raw_pre_alignment',
        'crop_seg': 'crop_seg_pre_alignment'
        }
    )

    if settings['continue_from_previous']:
        # In the event a previous run was aborted, e.g. due to no disk space
        # Recreate the cell_df as if we were mid run
        done_fovs = pd.read_csv(step_dir / 'manifest.csv')
        # Save old manifest in case I screwed up (can remove once I am sure it works properly)
        done_fovs.to_csv(step_dir / 'old_manifest.csv')
        not_done_fovs = cell_df[~cell_df['fov_id'].isin(done_fovs['fov_id_x'])]
        cell_df = pd.concat([done_fovs, not_done_fovs])
        fov_df = create_fov_dataframe_from_cell_dataframe(not_done_fovs)

    else:
        fov_df = create_fov_dataframe_from_cell_dataframe(cell_df)
    
    return cell_df, fov_df


def calculate_alignment_angles(
    img: np.ndarray, 
    mode: str, 
    use_channels: Union[int, Tuple],
    centroid_normed: Tuple
) -> Tuple:
    # Initialize angles at 0 because some methods only calcualte 1-2 angles
    angle_1, angle_2, angle_3 = (0, 0, 0)
    img_subsetted = img[(use_channels), :, :, :]

    if mode == 'unaligned':
        angle_1, angle_2, angle_3 = (0, 0, 0)

    if mode in ('xy_only', 'xy_xz', 'xy_xz_yz'):
        angle_1 = calculate_alignment_angle_2d(
                centroid_normed=centroid_normed,
        )
    
    if mode in ('xy_xz', 'xy_xz_yz'):
        angle_2 = calculate_2d_long_axis_angle_to_z_axis(
            img_subsetted,
            'xz'
        )
    
    if mode == 'xy_xz_yz':
        angle_3 = calculate_2d_long_axis_angle_to_z_axis(
            img_subsetted,
            'yz'
        )

    if mode == 'principal_axes':
        angle_1, angle_2, angle_3 = rotate_image_3d(
            img_subsetted
        )

    return angle_1, angle_2, angle_3


def execute_step(config):
    step_name = 'alignment'
    settings = get_alignment_settings(config)

    check_dir_exists(settings['project_dir'])
    step_dir = create_step_dir(settings['project_dir'], step_name)

    logger = step_logger(step_name, step_dir)
    logger.info(sys.argv)

    cell_df, fov_df = prepare_cell_and_fov_datasets(settings, step_dir)

    # Calculate neuromast centroid and rotation angles
    cell_angles = []

    for fov in fov_df.itertuples(index=False):

        current_fov_cells, fov_info, seg_img = prepare_fov(fov, settings, cell_df)
        print('fov preparation complete')

        for cell in current_fov_cells.itertuples(index=False):

            # Initialize a dict in which to store cell info
            cell_info = {}
            cell_info['nm_centroid'] = fov_info['nm_centroid']
            cell_info['fov_id'] = fov.fov_id
            cell_info['CellId'] = cell.CellId
            print('Aligning', cell.CellId)

            label = int(cell.label)

            cell_img = np.where(seg_img == label, 1, 0)

            # Calculate alignment angles
            cell_img = cell_img.astype(np.uint8)
            cell_img = cell_img * 255
            cell_centroid = ndi.center_of_mass(cell_img)
            cell_info['cell_centroid_normed'] = normalize_centroid(cell_img, cell_info['nm_centroid'])

            # Apply xy alignment to seg and raw crops
            raw_cell, seg_cell = read_raw_and_seg_img(cell.crop_raw_pre_alignment, cell.crop_seg_pre_alignment)

            try:
                # Initialize angles at 0 because some methods only calculate 1-2 angles
                mode = settings['mode']
                use_channels = settings['use_channels']
                centroid_normed = cell_info['cell_centroid_normed']
                angle_1, angle_2, angle_3 = (0, 0, 0)
                img_subsetted = seg_cell[(use_channels), :, :, :]

                if mode == 'unaligned':
                    angle_1, angle_2, angle_3 = (0, 0, 0)

                if mode in ('xy_only', 'xy_xz', 'xy_xz_yz'):
                    angle_1 = calculate_alignment_angle_2d(
                            centroid_normed=centroid_normed,
                    )
                    seg_cell_aligned = ndi.rotate(seg_cell, -angle_1, (2, 3), reshape=True, order=0)
                
                if mode in ('xy_xz', 'xy_xz_yz'):
                    angle_2 = calculate_2d_long_axis_angle_to_z_axis(
                        img_subsetted,
                        'xz'
                    )
                    seg_cell_aligned = ndi.rotate(seg_cell_aligned, angle_2, (1, 3), reshape=True, order=0)

                
                if mode == 'xy_xz_yz':
                    angle_3 = calculate_2d_long_axis_angle_to_z_axis(
                        img_subsetted,
                        'yz'
                    )
                    seg_cell_aligned = ndi.rotate(seg_cell_aligned, angle_3, (1, 2), reshape=True, order=0)


                if mode == 'principal_axes':
                    angle_1, angle_2, angle_3 = rotate_image_3d(
                        img_subsetted
                    )
                    seg_cell_aligned = apply_3d_rotation(seg_cell, angle_1, angle_2, angle_3)

                raw_cell_aligned = apply_3d_rotation(raw_cell, angle_1, angle_2, angle_3)

                cell_info['rotation_angle'] = angle_1
                cell_info['rotation_angle_2'] = angle_2
                cell_info['rotation_angle_3'] = angle_3
                cell_info['centroid'] = cell_centroid
            
            except (MemoryError, ValueError) as e:
                print(e)
                logger.info('For cell %s of %s, encountered error: %s',
                            label, fov.fov_id, e)
                continue

            else:
                current_cell_dir = f'{step_dir}/{fov.fov_id}/{label}'
                raw_path, seg_path = save_raw_and_seg_cell(raw_cell_aligned, seg_cell_aligned, current_cell_dir)

                cell_info['crop_raw'] = raw_path
                cell_info['crop_seg'] = seg_path

                # Save angle matched to cell_id
                # Also saves cell centroid and paths for rotated single cells
                cell_angles.append({**cell_info})

        # Save angles to cell manifest
        angle_df = pd.DataFrame(cell_angles)
        new_cell_df = cell_df.merge(angle_df, on='CellId')
        new_cell_df.to_csv(f'{step_dir}/manifest.csv')
        print('Manifest saved.')


def main():
    parser = argparse.ArgumentParser(description='Basic cell alignment')
    parser.add_argument('config', help='path to config file')
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.Loader)
    execute_step(config)


if __name__ == '__main__':
    main()
