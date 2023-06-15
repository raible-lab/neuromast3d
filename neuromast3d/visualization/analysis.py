#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Automate some of the downstream analysis (PCA, UMAP, clustering) and output 
an AnnData object containing the results.

Depending on the settings, images, meshes, and reconstructions may be also be 
output.
'''

import argparse
import logging
from pathlib import Path
import sys

import anndata as ad
from matplotlib import pyplot as plt
import napari
import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import external, pp, tl, pl
import seaborn as sns

from neuromast3d.misc.find_closest_cells import RepresentativeCellFinder
from neuromast3d.visualization import plotting_tools


# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main():

    # Take inputs here
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dirs', type=str, nargs='+')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--pca_batches', type=int, nargs='+')
    parser.add_argument('--pca_alias', type=str)
    parser.add_argument('--resolution', type=float)
    parser.add_argument('--classify_rec_error', action='store_true')

    args = parser.parse_args()

    project_dirs = [Path(pd) for pd in args.project_dirs]
    ls_dirs = [pd / 'local_staging' for pd in project_dirs]
    curation_files = [pd / 'curated_fov_dataset.csv' for pd in project_dirs]
    output_dir = Path(args.output_dir)
    pca_batches = args.pca_batches
    alias = args.pca_alias
    resolution = args.resolution
    classify_rec_error = args.classify_rec_error

    # Set up logging - file handler
    f_handler = logging.FileHandler(f'{output_dir}/analysis.log', mode='w')
    formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    # Set stream handler
    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.DEBUG)
    s_handler.setFormatter(formatter)
    logger.addHandler(s_handler)

    logger.info(sys.argv)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Needed so scanpy plots save in the right place
    sc._settings.ScanpyConfig.figdir = output_dir

    # Automatically combine experiments
    feat_data = plotting_tools.combine_features_datasets(ls_dirs)
    curated_feat_data = pd.concat([pd.read_csv(file) for file in curation_files])

    # HC col must be added before excluding cells (b/c they can overlap)
    feat_data['hair_cell'] = plotting_tools.get_hcs(feat_data, curated_feat_data)
    feat_data = plotting_tools.drop_manually_curated_cells(feat_data, curated_feat_data)
    feat_data['CellId'] = feat_data.index

    # TODO: why does the column need to be renamed here?
    # Seems like it got named 'fov_id_x' upstream...
    feat_data = feat_data.rename({'fov_id_x': 'fov_id'}, axis=1)
    feat_data = feat_data.merge(curated_feat_data, on='fov_id', how='left')
    feat_data = feat_data.set_index('CellId')

    # Remove cells based on size
    feat_data = feat_data.drop(feat_data.loc[feat_data['MEM_shape_volume'] < 400000].index)
    feat_data = feat_data.drop(feat_data.loc[feat_data['MEM_shape_volume'] > 3500000].index)

    # Spherical harmonics coeffs are a standin for gene expression here
    matrix_of_shcoeffs, feature_names = plotting_tools.get_matrix_of_shcoeffs_for_pca(feat_data, alias=alias)
    adata = ad.AnnData(matrix_of_shcoeffs)
    adata.obs_names = feat_data.index
    adata.var_names = feature_names
    print('adata size', adata.X.shape)
    print('df size', feat_data.shape)
    adata.obs['genotype'] = plotting_tools.get_genotypes_from_cellids(feat_data)
    adata.obs['hair_cell'] = feat_data['hair_cell']

    if classify_rec_error:
        logger.info('classifying cells by rec error')
        adata = plotting_tools.classify_rec_error(adata, ls_dirs, alias)

    adata.obs['batch'] = feat_data['batch']

    # TODO: do we still need this part? I think so...
    adata.obsm['other_features'] = feat_data[feat_data.columns.difference(feature_names)]
    cell_positions = plotting_tools.calculate_cell_positions(adata.obsm['other_features'])
    adata.obs = pd.concat([adata.obs, cell_positions], axis=1)

    # Analyze intensity based stuff
    intensity_cols = {}
    for ls_dir in ls_dirs:
        try:
            path_to_config = ls_dir / 'computefeatures/parameters.yaml'
            intensity_cols.update(
                plotting_tools.get_intensity_cols_from_config(path_to_config)
            )
        except KeyError as e:
            logger.info(e)
            logger.info(f'No intensity channel in {path_to_config}')

    adata.uns['intensity_cols'] = intensity_cols
    adata.obs['transgene'] = plotting_tools.get_transgenes(adata)

    # Calculate intensity z-scores
    for ch_name, name in intensity_cols.items():
        col_name = f'{name}_intensity_mean_lcc'
        adata.obs[f'{col_name}'] = adata.obsm['other_features'][col_name]
        adata.obs = plotting_tools.add_intensity_z_score_col(adata.obs, col_name, suffix=ch_name)
        binarized = adata.obs[f'{col_name}_z_score'] > 1

        # If not from an experiment with that label, remain NaN
        binarized[adata.obs[f'{col_name}_z_score'].isnull()] = np.NaN
        adata.obs[f'{ch_name}_positive'] = binarized.astype(float)

    # Fix columns that prevent saving
    #adata.obsp['pheno_jaccard_ig'] = adata.obsp['pheno_jaccard_ig'].tocsr()
    #adata.obsm['other_features']['nm_centroid'] = adata.obsm['other_features']['nm_centroid'].astype('str')
    adata.obs['nm_centroid'] = adata.obs['nm_centroid'].astype('str')
    adata.obsm['other_features']['polarity'] = adata.obsm['other_features']['polarity'].astype('str')
    adata.obsm['other_features']['centroid'] = adata.obsm['other_features']['centroid'].astype('str')
    adata.obs['cell_centroid'] = adata.obs['cell_centroid'].astype('str')

    # Don't know why this is needed all of the sudden...
    adata.obsm['other_features']['cells_to_exclude'] = adata.obsm['other_features']['cells_to_exclude'].astype('str')

    adata.write(output_dir / 'adata.h5ad')


if __name__ == '__main__':
    main()
