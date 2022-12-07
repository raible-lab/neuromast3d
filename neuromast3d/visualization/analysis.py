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
    parser.add_argument('--curation_csvs', type=str, nargs='+')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--pca_batches', type=int, nargs='+')
    parser.add_argument('--pca_alias', type=str)
    parser.add_argument('--resolution', type=float)
    parser.add_argument('--classify_rec_error', action='store_true')

    args = parser.parse_args()

    project_dirs = [Path(pd) for pd in args.project_dirs]
    curation_files = [Path(cf) for cf in args.curation_csvs]
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
    feat_data = plotting_tools.combine_features_datasets(project_dirs)
    curated_feat_data = pd.concat([pd.read_csv(file) for file in curation_files])
    feat_data = plotting_tools.drop_manually_curated_cells(feat_data, curated_feat_data)
    feat_data['CellId'] = feat_data.index

    # TODO: why does the column need to be renamed here?
    # Seems like it got named 'fov_id_x' upstream...
    feat_data = feat_data.rename({'fov_id_x': 'fov_id'}, axis=1)
    feat_data = feat_data.merge(curated_feat_data, on='fov_id', how='left')
    feat_data = feat_data.set_index('CellId')
    feat_data['genotype'] = plotting_tools.get_genotypes_from_cellids(feat_data)
    feat_data = feat_data.drop(feat_data.loc[feat_data['MEM_shape_volume'] < 400000].index)

    # Spherical harmonics coeffs are a standin for gene expression here
    matrix_of_shcoeffs, feature_names = plotting_tools.get_matrix_of_shcoeffs_for_pca(feat_data, alias=alias)
    adata = ad.AnnData(matrix_of_shcoeffs)
    adata.obs_names = feat_data.index
    adata.var_names = feature_names

    if classify_rec_error:
        logger.info('classifying cells by rec error')
        adata = plotting_tools.classify_rec_error(adata, project_dirs, alias)
        sns.kdeplot(adata.obsm['rec_error'], x='max_hd', hue='gmm_classes')
        plt.tight_layout()
        plt.savefig(output_dir / 'rec_error_gmm_split.png')

    adata, pca, axes = plotting_tools.run_custom_pca(feat_data, adata, 'batch', pca_batches, alias, 0.90)
    logger.info(f'PCA done on batches {pca_batches}')

    num_pcs = len(axes.columns)
    logger.info(f'Number of PCs is: {num_pcs}')

    # Cluster, find neighbors, and UMAP embedding
    external.tl.phenograph(adata, clustering_algo='leiden', k=30, resolution_parameter=resolution, seed=42)
    pp.neighbors(adata, metric='minkowski', n_neighbors=30, n_pcs=num_pcs, random_state=42)

    # Generate PCA plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
    plotting_tools.plot_pca_var_explained(adata, ax, output_dir / 'pca_var_explained.png')

    # UMAP embedding
    fig, ax = plt.subplots(figsize=(8, 6))
    tl.umap(adata, random_state=42)
    pl.umap(adata, color='pheno_leiden', ax=ax, save='.svg', show=False)

    # Rearrange clusters by distance from neuromast center
    adata.obsm['other_features'] = feat_data[feat_data.columns.difference(feature_names)]
    adata.obsm['other_features'] = plotting_tools.add_distances_to_dataframe(adata.obsm['other_features'])

    adata.obsm['other_features']['pheno_leiden'] = adata.obs['pheno_leiden']
    adata.obs['pheno_leiden_sorted'] = plotting_tools.reorder_clusters(
        adata.obsm['other_features'], 
        'normalized_xy_dist_from_center', 
        'pheno_leiden', 
        'median', 
        False
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    tl.umap(adata, random_state=42)
    pl.umap(adata, color='pheno_leiden_sorted', ax=ax, save='.svg', show=False)

    # PAGA trajectory analysis
    tl.paga(adata, groups='pheno_leiden_sorted')
    fig, ax = plt.subplots(figsize=(7, 6))
    pl.paga(adata, threshold=0.2, ax=ax, title='PAGA trajectory analysis', color='pheno_leiden_sorted', save='_thresh0.2.svg', show=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    tl.umap(adata, init_pos='paga')
    pl.umap(adata, color='pheno_leiden_sorted', save='_init_pos_paga.svg', ax=ax, show=False)

    plotting_tools.reconstruct_pc_meshes(adata, pca, alias, 8, [-1, -0.5, 0, 0.5, 1], output_dir)

    # Find representative cells
    finder = RepresentativeCellFinder(adata.obsm['X_pca'])
    clust_centroids = finder.find_cluster_centroids(adata.obs['pheno_leiden_sorted'].values)
    repr_cells = finder.find_cells_near_all_cluster_centroids(1)
    repr_cells_df = pd.DataFrame(repr_cells)
    adata.uns['repr_cells'] = repr_cells_df

    fig, ax = plt.subplots(figsize=(8, 6))
    plotting_tools.plot_repr_cells_umap(adata, ax, output_dir / 'repr_cells_umap.png', hue=adata.obs['pheno_leiden_sorted'])

    viewer = plotting_tools.view_repr_cells(adata, 'pheno_leiden_sorted', output_dir)
    napari.run()

    plotting_tools.reconstruct_repr_cells_from_shcoeffs(adata, alias, output_dir)

    if len(project_dirs) > 1:
        # Multiple batches, plot comparison
        fig, ax = plt.subplots(figsize=(8, 6))
        plotting_tools.plot_umap_from_adata(adata, ax, output_dir / 'batch_umap.png', hue=adata.obsm['other_features']['batch'], palette='deep')

    # Polar plots of cell locations
    mpl_deep = plotting_tools.convert_seaborn_cmap_to_mpl('deep')
    adata.obsm['other_features']['pheno_leiden_sorted'] = adata.obs['pheno_leiden_sorted']
    plotting_tools.plot_clusters_polar(adata.obsm['other_features'], 'pheno_leiden_sorted', mpl_deep)
    plt.savefig(output_dir / 'polar_plots_normalized.png')

    # Analyze intensity based stuff
    intensity_cols = {}
    for proj_dir in project_dirs:
        path_to_config = proj_dir / 'computefeatures/parameters.yaml'
        intensity_cols.update(
            plotting_tools.get_intensity_cols_from_config(path_to_config)
        )

    # Calculate intensity z-scores and make intensity plots
    for ch_name, col_name in intensity_cols.items():
        adata.obsm['other_features'] = plotting_tools.add_intensity_z_score_col(adata.obsm['other_features'], col_name, suffix=ch_name)
        adata.obsm['other_features'][f'{ch_name}_positive'] = np.where(adata.obsm['other_features'][f'{col_name}_z_score'] > 1, 1, 0)
        plotting_tools.plot_intensity_umap(adata, ch_name, col_name)
        plt.savefig(output_dir / f'{ch_name}_intensity_umap.png', dpi=300)


    # Fix columns that prevent saving
    adata.obsp['pheno_jaccard_ig'] = adata.obsp['pheno_jaccard_ig'].tocsr()
    adata.obsm['other_features']['nm_centroid'] = adata.obsm['other_features']['nm_centroid'].astype('str')
    adata.obsm['other_features']['polairty'] = adata.obsm['other_features']['polairty'].astype('str')
    adata.obsm['other_features']['cell_centroid'] = adata.obsm['other_features']['cell_centroid'].astype('str')

    adata.write(output_dir / 'adata.h5ad')
    plt.show()


if __name__ == '__main__':
    main()
