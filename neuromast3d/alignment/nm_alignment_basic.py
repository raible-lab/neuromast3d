#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Align all cells to same cardinal direction

As suggested by Lorenzo, this script assumes radial symmetry of the neuromast
and does not try to align cells to an organismal axis (e.g. A/P, D/V).
"""

import ast
import argparse
import logging
import os
import pathlib
import sys

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
from aicsimageprocessing import resize, resize_to
import numpy as np
import pandas as pd
from scipy.stats import skew
from skimage.morphology import binary_closing, ball
from skimage.measure import regionprops
from skimage.transform import rotate
from sklearn.decomposition import PCA
import scipy.ndimage as ndi
import yaml

from neuromast3d.prep_single_cells.create_fov_dataset import read_raw_and_seg_img, check_dir_exists, step_logger
from neuromast3d.alignment.utils import rotate_image_2d_custom


def calculate_alignment_angle_2d(
        image: np.array,
        origin: tuple,
        make_unique: bool = True
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
    -------
    Tuple[float, np.array]
        The angle for rotation and the image centroid after subtracting
        the defined origin. The sign of the angle can be positive or negative,
        depending on where the cell centroid was relative to the origin.
        Cells with positive angles should be rotated CW, and those with
        negative angles should be rotated CCW.

    """

    if image.ndim != 3:
        raise ValueError(f'Invalid shape of input image {image.shape}. \
                Image must be a single-channel, 3D stack.')

    centroid = ndi.center_of_mass(image)
    centroid_normed = np.subtract(centroid, origin)
    x = centroid_normed[2]
    y = -centroid_normed[1]

    if make_unique:  # NOTE: modified from original in shparam!

        # Calculate angle with atan2 to preserve orientation
        # I think this SHOULD align to the 3 o' clock position?
        angle = 180 * np.arctan2(y, x) / np.pi

    else:
        # TODO: update or remove? Don't think this is right anymore
        # Calculate smallest angle
        angle = 0.0
        if np.abs(centroid[2]) > 1e-12:  # avoid divide by zero error ig?
            angle = 180 * np.arctan(centroid[1] / centroid[2]) / np.pi

    return angle, centroid


def align_cell_xy_long_axis_to_z_axis(raw_cell, seg_cell):
    _, z, y, x = np.nonzero(seg_cell)
    xz = np.hstack([x.reshape(-1, 1), z.reshape(-1, 1)])
    pca = PCA(n_components=2)
    pca = pca.fit(xz)
    eigenvecs = pca.components_
    angle = 180 * np.arctan(eigenvecs[0][0]/eigenvecs[0][1]) / np.pi
    seg_cell_aligned = ndi.rotate(seg_cell, -angle, axes=(1, 3), order=0)
    raw_cell_aligned = ndi.rotate(raw_cell, -angle, axes=(1, 3), order=0)
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


def save_raw_and_seg_cell(raw_img, seg_img, current_cell_dir):
    pathlib.Path(current_cell_dir).mkdir(parents=True, exist_ok=True)
    seg_path = f'{current_cell_dir}/segmentation.ome.tif'
    crop_seg_aligned_path = pathlib.path(seg_path)
    writer = ome_tiff_writer.OmeTiffWriter(crop_seg_aligned_path)
    writer.save(seg_img, dimension_order='czyx')

    raw_path = f'{current_cell_dir}/raw.ome.tif'
    crop_raw_aligned_path = pathlib.Path(raw_path)
    writer = ome_tiff_writer.OmeTiffWriter(crop_raw_aligned_path)
    writer.save(raw_img, dimension_order='CZYX')
    return raw_path, seg_path


def execute_step(config):
    step_name = 'alignment'
    project_dir = pathlib.Path(config['create_fov_dataset']['output_dir'])
    path_to_manifest = project_dir / 'prep_single_cells/cell_manifest.csv'
    rot_ch_index = config['alignment']['rot_ch_index']
    make_unique = config['alignment']['make_unique']
    mode = config['alignment']['mode']

    check_dir_exists(project_dir)

    # Read the manifest to align cells for this run
    cell_df = pd.read_csv(path_to_manifest, index_col=0)

    # Since we are applying alignment, rename old crop_seg and crop_raw
    # Because we want to use the aligned images in future steps
    cell_df = cell_df.rename(columns={
        'crop_raw': 'crop_raw_pre_alignment',
        'crop_seg': 'crop_seg_pre_alignment'
        }
    )

    # Create dir to save for this step
    step_local_path = pathlib.Path(f'{project_dir}/alignment')
    step_local_path.mkdir(parents=True, exist_ok=True)

    # Save command line arguments to logfile for future reference
    logger = step_logger(step_name, step_local_path)
    logger.info(sys.argv)

    # Create fov dataframe
    fov_df = create_fov_dataframe_from_cell_dataframe(cell_df)

    # Calculate neuromast centroid and rotation angles
    nm_centroids = []
    cell_angles = []

    # Loop through and interpolate each neuromast (fov)
    for fov in fov_df.itertuples(index=False):
        seg_reader = AICSImage(fov.fov_seg_path)

        # TODO: For now, nm and cell centroid calculation will just use the
        # membrane channel. But it could be useful to have an option
        # to rotate about the centroid calculated from the nucleus channel.
        seg_img = seg_reader.get_image_data('ZYX', S=0, T=0, C=rot_ch_index)
        nm = seg_img > 0
        pixel_size_xyz = ast.literal_eval(fov.pixel_size_xyz)
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

        # Save nm centroids matched to fov_id
        nm_centroids.append({'fov_id': fov.fov_id, 'nm_centroid': nm_centroid})

        # Subset cell df for this fov
        current_fov_cells = cell_df[cell_df['fov_id'] == fov.fov_id]

        for cell in current_fov_cells.itertuples(index=False):

            label = int(cell.label)
            cell_img = np.where(seg_img == label, seg_img, 0)

            # Calculate alignment angle in xy plane
            cell_img = cell_img.astype(np.uint8)
            cell_img = cell_img * 255
            rotation_angle, cell_centroid = calculate_alignment_angle_2d(
                    image=cell_img,
                    origin=nm_centroid,
                    make_unique=make_unique
            )

            # Apply xy alignment to seg and raw crops
            raw_cell, seg_cell = read_raw_and_seg_img(cell.crop_raw_pre_alignment, cell.crop_seg_pre_alignment)

            # Rotate function expects multichannel image
            if seg_cell.ndim == 3:
                seg_cell = np.expand_dims(seg_cell, axis=0)
            seg_cell_aligned = rotate_image_2d_custom(
                    image=seg_cell,
                    angle=rotation_angle,
                    interpolation_order=0,
                    flip_angle_sign=True
            )

            if raw_cell.ndim == 3:
                raw_cell = np.expand_dims(raw_cell, axis=0)
            raw_cell_aligned = rotate_image_2d_custom(
                    image=raw_cell,
                    angle=rotation_angle,
                    interpolation_order=0,
                    flip_angle_sign=True
            )

            if mode == 'xy_xz':
                # Do an additional rotation to align xy long axis to z axis
                # TODO: save this angle too?
                raw_cell_aligned, seg_cell_aligned = align_cell_xy_long_axis_to_z_axis(raw_cell_aligned, seg_cell_aligned)

            # Save aligned single cell mask and raw image
            current_cell_dir = f'{step_local_path}/{fov.fov_id}/{label}'
            raw_path, seg_path = save_raw_and_seg_cell(raw_cell_aligned, seg_cell_aligned, current_cell_dir)

            # Save angle matched to cell_id
            # Also saves cell centroid and paths for rotated single cells
            cell_angles.append({
                'CellId': cell.CellId,
                'rotation_angle': rotation_angle,
                'nm_centroid': nm_centroid,
                'centroid': cell_centroid,
                'crop_raw': raw_path,
                'crop_seg': seg_path
            })

    # Add to cell_df
    fov_centroid_df = pd.DataFrame(nm_centroids)
    cell_df = cell_df.merge(fov_centroid_df, on='fov_id')

    # Save angles to cell manifest
    angle_df = pd.DataFrame(cell_angles)
    cell_df = cell_df.merge(angle_df, on='CellId')
    cell_df.to_csv(f'{step_local_path}/manifest.csv')


def main():
    parser = argparse.ArgumentParser(description='Basic cell alignment')
    parser.add_argument('config', help='path to config file')
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.Loader)
    execute_step(config)


if __name__ == '__main__':
    main()
