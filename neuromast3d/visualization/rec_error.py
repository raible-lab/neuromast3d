#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Calculate several reconstruction error metrics for spherical harmonics 
representations of a given dataset.

Input: results from cvapipe_analysis computefeatures step
Output: csv file with measured errors for each CellId
'''

import argparse
import ast
from pathlib import Path

from aicsimageio import AICSImage
from aicsshparam import shtools
import numpy as np
import pandas as pd
from tqdm import tqdm
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from neuromast3d.visualization.plotting_tools import get_features_data, get_matrix_of_shcoeffs_for_pca
from neuromast3d.visualization.analysis import reconstruct_mesh_from_shcoeffs_array


def get_distances_between_meshes(original, reconstructed):
    coords_orig = vtk_to_numpy(original.GetPoints().GetData())
    coords_rec = vtk_to_numpy(reconstructed.GetPoints().GetData())

    # Calculate forward distances
    Tree = vtk.vtkKdTreePointLocator()
    Tree.SetDataSet(original)
    Tree.BuildLocator()

    forward_dists = []
    for i in range(coords_rec.shape[0]):
        j = Tree.FindClosestPoint(coords_rec[i])
        dist = np.linalg.norm(coords_rec[i]-coords_orig[j])
        forward_dists.append(dist)

    forward_dists = np.array(forward_dists)

    # Calculate reverse distances
    Tree = vtk.vtkKdTreePointLocator()
    Tree.SetDataSet(reconstructed)
    Tree.BuildLocator()

    reverse_dists = []
    for i in range(coords_orig.shape[0]):
        j = Tree.FindClosestPoint(coords_orig[i])
        dist = np.linalg.norm(coords_orig[i]-coords_rec[j])
        reverse_dists.append(dist)

    reverse_dists = np.array(reverse_dists)
    return forward_dists, reverse_dists


def calculate_hausdorff_distances(forward_dists, reverse_dists):
    forward_hd = np.max(forward_dists)
    reverse_hd = np.max(reverse_dists)
    max_hd = max(forward_hd, reverse_hd)
    return forward_hd, reverse_hd, max_hd


def calculate_mean_dist_error(forward_dists, reverse_dists):
    forward_mde = forward_dists.mean()
    reverse_mde = reverse_dists.mean()
    return forward_mde, reverse_mde


def calculate_std_dist_error(forward_dists, reverse_dists):
    forward_sde = forward_dists.std()
    reverse_sde = reverse_dists.std()
    return forward_sde, reverse_sde


def measure_reconstruction_error(original, reconstructed, scale):
    forward_dists, reverse_dists = get_distances_between_meshes(original, reconstructed)
    forward_hd, reverse_hd, max_hd = calculate_hausdorff_distances(forward_dists, reverse_dists)
    forward_mde, reverse_mde = calculate_mean_dist_error(forward_dists, reverse_dists)
    forward_sde, reverse_sde = calculate_std_dist_error(forward_dists, reverse_dists)
    errors = {
        'forward_hd': forward_hd * scale,
        'reverse_hd': reverse_hd * scale,
        'max_hd': max_hd * scale,
        'forward_mde': forward_mde * scale,
        'reverse_mde': reverse_mde * scale,
        'forward_sde': forward_sde * scale,
        'reverse_sde': reverse_sde * scale,
    }
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('local_staging_dir')
    parser.add_argument('alias')

    args = parser.parse_args()
    ls_dir = Path(args.local_staging_dir)
    alias = str(args.alias)

    features_data = get_features_data(ls_dir)
    shcoeffs, feat_names = get_matrix_of_shcoeffs_for_pca(features_data, alias=alias)

    rec_errors = []

    rec_errors = []
    for row_ind, cell_shcoeffs in tqdm(enumerate(shcoeffs)):
        coeffs, mesh_rec, grid_rec = reconstruct_mesh_from_shcoeffs_array(
                cell_shcoeffs,
                feat_names,
                alias,
                32,
                save_path=None
            )
        seg_path = features_data['crop_seg'][row_ind]

        reader = AICSImage(seg_path)
        seg_img = reader.get_image_data('ZYX', C=0, S=0, T=0)
        mesh, _, _ = shtools.get_mesh_from_image(seg_img)

        pixel_size = ast.literal_eval(features_data['pixel_size_xyz'][row_ind])
        scale_factor = pixel_size[0]*10**6
        errors = measure_reconstruction_error(mesh, mesh_rec, scale_factor)
        errors['CellId'] = features_data.index[row_ind]
        rec_errors.append(errors)

    rec_errors = pd.DataFrame.from_dict(rec_errors)

    rec_errors.to_csv(ls_dir / f'rec_errors_{alias}.csv')

if __name__ == '__main__':
    main()
