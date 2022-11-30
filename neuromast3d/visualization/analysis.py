#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Automate some of the downstream analysis (PCA, UMAP, clustering) and output 
an AnnData object containing the results.

Depending on the settings, images, meshes, and reconstructions may be also be 
output.
'''

import argparse
import ast
import logging
from pathlib import Path
import sys
from typing import Optional, Union

from aicsimageio import AICSImage
from aicsshparam import shtools
import anndata as ad
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
import napari
import numpy as np
import pandas as pd
import phenograph
import scanpy as sc
from scanpy import external, pp, tl, pl
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tifffile import imread, imsave
import umap
import yaml

from neuromast3d.misc.find_closest_cells import RepresentativeCellFinder
from neuromast3d.visualization import plotting_tools


# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_mesh_from_dict(row: dict, alias: str, lmax: int):
    cos_coeffs = np.zeros((1, lmax, lmax), dtype=np.float32)
    sin_coeffs = np.zeros((1, lmax, lmax), dtype=np.float32)
    for l in range(lmax):
        for m in range(l + 1):
            try:
                cos_coeffs[0, l, m] = row[f'{alias}_shcoeffs_L{l}M{m}C_lcc']
                sin_coeffs[0, l, m] = row[f'{alias}_shcoeffs_L{l}M{m}S_lcc']
            except ValueError:
                pass
    coeffs = np.concatenate((cos_coeffs, sin_coeffs), axis=0)
    mesh, grid = shtools.get_reconstruction_from_coeffs(coeffs)
    return coeffs, mesh, grid


def reconstruct_mesh_from_shcoeffs_array(
    shcoeffs_vals: np.ndarray,
    shcoeffs_names: pd.Index,
    alias: str,
    lmax: int,
    save_path: Optional[str] = None
):
    row_dict = {index: shcoeffs_vals[count] for count, index in enumerate(shcoeffs_names)}
    coeffs, mesh, grid = get_mesh_from_dict(row_dict, alias, lmax)
    if save_path is not None:
        shtools.save_polydata(mesh, save_path)
    return coeffs, mesh, grid



def reconstruct_pc_meshes(adata, pca_model, alias: str, num_pcs: int, sd_bins: list, output_dir):
    pc_means = adata.obsm['X_pca'].mean(axis=0)
    pc_sds = adata.obsm['X_pca'].std(axis=0)
    for pc in range(num_pcs + 1):
        for n_sd in sd_bins:
            means_tweaked = pc_means.copy()
            means_tweaked[pc] = pc_means[pc] + (n_sd * pc_sds[pc])
            shcoeffs = np.array(pca_model.inverse_transform(means_tweaked))
            mesh_dir = Path(output_dir / 'PC_mesh_representations')
            mesh_dir.mkdir(parents=True, exist_ok=True)
            save_path = f'{mesh_dir}/{pc}_{n_sd}_mesh.vtk'
            reconstruct_mesh_from_shcoeffs_array(shcoeffs, adata.var_names, alias, 32, save_path)
    return


def run_custom_pca(
    df,
    adata,
    subset_col: str,
    subset_vals: list,
    alias: str,
    n_comps: Union[float, int],
    zero_center: Optional[bool] = True,
    use_highly_variable: Optional[bool] = None,
):
    # Fit PCA to subset of data, apply transform to rest, then add to adata object
    # Modified from original scanpy function
    pca_ = plotting_tools.fit_pca_to_subset(df, subset_col, subset_vals, alias, n_comps)
    axes = plotting_tools.apply_pca_transform(df, pca_, alias=alias)
    X_pca = axes.values
    adata.obsm['X_pca'] = X_pca
    adata.uns['pca'] = {}
    adata.uns['pca']['params'] = {
        'zero_center': zero_center,
        'use_highly_variable': use_highly_variable,
    }

    if use_highly_variable:
        adata.varm['PCs'] = np.zeros(shape=(adata.n_vars, n_comps))
        adata.varm['PCs'][adata.var['highly_variable']] = pca_.components_.T

    else:
        adata.varm['PCs'] = pca_.components_.T

    adata.uns['pca']['variance'] = pca_.explained_variance_
    adata.uns['pca']['variance_ratio'] = pca_.explained_variance_ratio_

    return adata, pca_, axes


def get_cluster_percents_by_genotype(
    df_clustered: pd.DataFrame,
    cluster_name: str,
    order: Optional[list] = None
):
    cluster_percents = pd.DataFrame()
    for gt in np.unique([df_clustered['genotype'].values]):
        clust_percents = df_clustered.loc[df_clustered['genotype'] == gt][cluster_name].value_counts(normalize=True)
        clust_percents = pd.DataFrame(clust_percents)
        clust_percents['genotype'] = gt
        cluster_percents = pd.concat([cluster_percents, clust_percents])
    cluster_percents['percent_per_cluster'] = cluster_percents[cluster_name]
    cluster_percents = cluster_percents.drop(cluster_name, axis=1)
    cluster_percents[cluster_name] = cluster_percents.index
    if order is not None:
        cluster_percents['genotype'] = pd.Categorical(cluster_percents['genotype'], order)
        cluster_percents.sort_values(by='genotype')
    return cluster_percents


def add_distances_to_dataframe(df):
    df_merged = df.copy()
    # Calculate cell centroid dist
    df_merged['nm_centroid'] = df_merged['nm_centroid'].apply(ast.literal_eval)
    df_merged['cell_centroid'] = df_merged['centroid'].apply(ast.literal_eval)

    # Is this the true distance? Or do I need to square/square root it...
    dist = df_merged['cell_centroid'].apply(np.array) - df_merged['nm_centroid'].apply(np.array)

    # Separate 3 coordinates into separate columns
    df_merged['z_dist'], df_merged['y_dist'], df_merged['x_dist'] = zip(*dist)

    # Invert sign of y to convert from rc coords to xy coords'
    df_merged['y_dist'] = -df_merged['y_dist']

    # Drop z to only calculate xy dist
    xy_dist = df_merged[['y_dist', 'x_dist']].values.tolist()
    df_merged['raw_xy_dist_from_center'] = [np.linalg.norm(row) for row in xy_dist]

    # Convert rotation angle to rads for plotting
    df_merged['rotation_angle_in_rads'] = df_merged['rotation_angle']*np.pi/180

    # Will need to check this is right later, just doing it quickly for now
    # DV should be counterclockwise rotated by 90 degrees
    df_merged['corrected_rotation_angle'] = df_merged['rotation_angle']
    df_merged.loc[df_merged['polairty'] == 'DV', 'corrected_rotation_angle'] = df_merged['rotation_angle'] + 90

    df_merged['corrected_rotation_angle_in_rads'] = df_merged['corrected_rotation_angle']*np.pi/180

    # Normalize to centroid of cell furthest from the neuromast center
    groups = df_merged.groupby(['fov_id'])
    max_vals = groups.transform('max')
    df_merged['normalized_xy_dist_from_center'] = df_merged['raw_xy_dist_from_center'] / max_vals['raw_xy_dist_from_center']
    return df_merged


def convert_seaborn_cmap_to_mpl(sns_cmap):
    mpl_cmap = ListedColormap(sns.color_palette(sns_cmap).as_hex())
    return mpl_cmap


def plot_clusters_polar(df, col_name: str, cmap):
    clust_groups = df.groupby(col_name)
    fig, axs = plt.subplots(ncols=len(clust_groups), figsize=(20, 8), subplot_kw={'projection': 'polar'}, sharey=True)
    for name, group in clust_groups:
        # this assumes clusters are integers from 0 to n clusters
        axs[name].plot(
                group['corrected_rotation_angle_in_rads'].values,
                group['normalized_xy_dist_from_center'].values,
                '.',
                color=cmap(name)
            )
    plt.tight_layout()
    return axs


def plot_pca_var_explained(adata, ax):
    ax[0].plot(adata.uns['pca']['variance_ratio'])
    ax[0].set_xlabel('Number of principal components')
    ax[0].set_ylabel('Variance explained')
    ax[1].plot(np.cumsum(adata.uns['pca']['variance_ratio']))
    ax[1].set_xlabel('Number of PCs')
    ax[1].set_ylabel('Cumulative variance explained')
    ax[0].set_xticks(np.arange(0, adata.obsm['X_pca'].shape[1]+1, 10))
    ax[1].set_xticks(np.arange(0, adata.obsm['X_pca'].shape[1]+1, 10))
    return ax


def plot_repr_cells_umap(adata, col_name, ax):
    UMAP_1 = adata.obsm['X_umap'][:, 0]
    UMAP_2 = adata.obsm['X_umap'][:, 1]

    g = sns.scatterplot(x=UMAP_1, y=UMAP_2, hue=adata.obs[col_name], palette='deep')
    repr_cells_df = adata.uns['repr_cells']
    plt.scatter(UMAP_1[repr_cells_df['inds']], UMAP_2[repr_cells_df['inds']], s=70, color='black', edgecolors='white')
    plt.legend(title='cluster', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    g.set_xlabel('UMAP 1')
    g.set_ylabel('UMAP 2')
    return ax


def plot_batch_umap(adata, ax):
    UMAP_1 = adata.obsm['X_umap'][:, 0]
    UMAP_2 = adata.obsm['X_umap'][:, 1]

    g = sns.scatterplot(x=UMAP_1, y=UMAP_2, hue=adata.obsm['other_features']['batch'], palette='deep')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    g.set_xlabel('UMAP 1')
    g.set_ylabel('UMAP 2')
    return ax


def view_repr_cells(adata, col_name, output_dir = None):
    custom_colors = {0: '#000000', 1: '#4c72b0', 2: '#dd8452', 3: '#55a868', 4: '#c44e52', 5: '#8172b3', 6: '#937860', 7:'#da8bc3', 8:'#8c8c8c', 9:'#ccb974', 10:'#64b5cd'}

    viewer = napari.Viewer()
    for count, ind in enumerate(adata.uns['repr_cells']['inds']):
        path_to_seg = adata.obsm['other_features'].iloc[ind].crop_seg
        seg_img = imread(path_to_seg)
        img_ch0 = seg_img[0, :, :, :]
        img_ch1 = seg_img[1, :, :, :]

        cluster = adata.obs[col_name].iloc[ind]
        img_ch0 = np.where(img_ch0 > 0, cluster + 1, 0)
        img_ch1 = np.where(img_ch1 > 0, cluster + 1, 0)

        viewer.add_labels(img_ch0, name=f'{cluster}_ch0', blending='additive', color=custom_colors)
        viewer.add_labels(img_ch1, name=f'{cluster}_ch1', blending='additive', color=custom_colors)
 
        if output_dir is not None:
            save_path = Path(output_dir / 'repr_cells')
            save_path.mkdir(parents=True, exist_ok=True)
            imsave(save_path / f'{cluster}_{count}.tiff', seg_img)

    return viewer


def reconstruct_repr_cells_from_shcoeffs(adata, alias, output_dir):
    # TODO: would be nice to save NUC too?
    for row in adata.uns['repr_cells'].itertuples():
        shcoeffs = adata.X[row.inds, :]
        save_dir = Path(output_dir / 'repr_cell_meshes')
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = f'{save_dir}/{row.cluster}_{row.k}_{alias}_reconstructed.vtk'
        reconstruct_mesh_from_shcoeffs_array(shcoeffs, adata.var_names, alias, 32, save_path)

        seg_path = adata.obsm['other_features']['crop_seg'][row.inds]
        reader = AICSImage(seg_path)
        seg_img = reader.get_image_data('ZYX', C=1, S=0, T=0)

        mesh, _, _ = shtools.get_mesh_from_image(seg_img)
        save_path = f'{save_dir}/{row.cluster}_{row.k}_original.vtk'
        shtools.save_polydata(mesh, save_path)


def get_intensity_cols_from_config(path_to_config):
    with open(path_to_config, 'r') as stream:
        params = yaml.safe_load(stream)
    # some configs may not have "intensity" settings
    raw_aliases = list(params['features']['intensity'].keys())
    intensity_col_names = {}
    for alias in raw_aliases:
        for key, val in params['data'].items():
            if alias in val.values():
                intensity_col_names[val['channel']] = f'{alias}_intensity_mean_lcc' 
    return intensity_col_names


def plot_intensity_umap(adata, ch_name, int_col):
    plt_args = plt_args = {'edgecolor': None, 's': 70}
    fig, ax = plt.subplots(figsize=(20, 7), ncols=2)
    subset = adata[adata.obsm['other_features'][ch_name].notna()]
    UMAP_1 = subset.obsm['X_umap'][:, 0]
    UMAP_2 = subset.obsm['X_umap'][:, 1]
    g = sns.scatterplot(data=subset.obsm['other_features'], x=UMAP_1, y=UMAP_2,
                        hue=int_col, palette='viridis', alpha=0.8, **plt_args, ax=ax[0])
    g2 = sns.scatterplot(data=subset.obsm['other_features'], x=UMAP_1, y=UMAP_2,
                         hue=f'{ch_name}_positive', palette=['grey', 'red'], alpha=0.8, **plt_args, ax=ax[1])
    ax[0].legend(title='intensity z-score', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    ax[1].legend(title=f'{ch_name} positive', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    g.set_xlabel('UMAP 1')
    g.set_ylabel('UMAP 2')
    g2.set_xlabel('UMAP 1')
    g2.set_ylabel('UMAP 2')
    plt.tight_layout()


def main():

    # Take inputs here
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dirs', type=str, nargs='+')
    parser.add_argument('--curation_csvs', type=str, nargs='+')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--pca_batches', type=int, nargs='+')
    parser.add_argument('--pca_alias', type=str)
    parser.add_argument('--resolution', type=float)
    parser.add_argument('--remove_bad_recs', action='store_true')

    args = parser.parse_args()

    project_dirs = [Path(pd) for pd in args.project_dirs]
    paths_to_curation_csvs = [Path(cc) for cc in args.curation_csvs]
    output_dir = Path(args.output_dir)
    pca_batches = args.pca_batches
    pca_alias = args.pca_alias
    resolution = args.resolution
    remove_bad_recs = args.remove_bad_recs

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

    if len(project_dirs) > 1:
        # Automatically combine experiments
        df = plotting_tools.combine_features_datasets(project_dirs, local_staging=False)
        curated_df = pd.DataFrame()
        for path in paths_to_curation_csvs:
            curated_df_new = pd.read_csv(path)
            curated_df = pd.concat([curated_df, curated_df_new])

    else:
        # Use data from only one experiment
        df = plotting_tools.get_features_data(project_dirs[0])
        df['batch'] = 1
        curated_df = pd.read_csv(paths_to_curation_csvs[0])

    df = plotting_tools.drop_manually_curated_cells(df, curated_df)
    df['CellId'] = df.index

    # TODO: why does the column need to be renamed here?
    # Seems like it got named 'fov_id_x' upstream...
    df = df.rename({'fov_id_x': 'fov_id'}, axis=1)
    df = df.merge(curated_df, on='fov_id', how='left')
    df = df.set_index('CellId')
    df['genotype'] = plotting_tools.get_genotypes_from_cellids(df)
    df = df.drop(df.loc[df['MEM_shape_volume'] < 400000].index)

    # Spherical harmonics coeffs are a standin for gene expression here
    matrix_of_shcoeffs, feature_names = plotting_tools.get_matrix_of_shcoeffs_for_pca(df, alias=pca_alias)
    adata = ad.AnnData(matrix_of_shcoeffs)
    adata.obs_names = df.index
    adata.var_names = feature_names

    if remove_bad_recs:
        logger.info('removing bad recs')
        for proj_dir in project_dirs:
            rec_errors = pd.read_csv(proj_dir / f'rec_errors_{pca_alias}.csv', index_col='CellId')
            rec_errors = rec_errors[rec_errors.index.isin(adata.obs_names)]
            adata.obsm['rec_error'] = rec_errors
            # Consider moving this to the error analysis script
            gm_model = GaussianMixture(n_components=2, covariance_type='tied')
            gm_model.fit(rec_errors['max_hd'].values.reshape(-1, 1))
            adata.obsm['rec_error']['gmm_classes'] = gm_model.predict(rec_errors['max_hd'].values.reshape(-1, 1))
            sns.kdeplot(adata.obsm['rec_error'], x='max_hd', hue='gmm_classes')
            plt.tight_layout()
            plt.savefig(output_dir / 'rec_error_gmm_split.png')

            adata = adata[adata.obsm['rec_error']['gmm_classes'] == 0]
            df = df.loc[adata.obs_names]

    adata, pca, axes = run_custom_pca(df, adata, 'batch', pca_batches, pca_alias, 0.90)
    logger.info(f'PCA done on batches {pca_batches}')

    num_pcs = len(axes.columns)
    logger.info(f'Number of PCs is: {num_pcs}')

    # Cluster, find neighbors, and UMAP embedding
    external.tl.phenograph(adata, clustering_algo='leiden', k=30, resolution_parameter=resolution, seed=42)
    pp.neighbors(adata, metric='minkowski', n_neighbors=30, n_pcs=num_pcs, random_state=42)

    # Generate PCA plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
    plot_pca_var_explained(adata, ax)
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_var_explained.png')

    # UMAP embedding
    fig, ax = plt.subplots(figsize=(8, 6))
    tl.umap(adata, random_state=42)
    pl.umap(adata, color='pheno_leiden', ax=ax, save='.svg', show=False)

    # Rearrange clusters by distance from neuromast center
    adata.obsm['other_features'] = df[df.columns.difference(feature_names)]
    adata.obsm['other_features'] = add_distances_to_dataframe(adata.obsm['other_features'])
    adata.obsm['other_features']['cluster'] = adata.obs['pheno_leiden']
    medians_per_clust = adata.obsm['other_features'].groupby('cluster').median(numeric_only=True)
    sds_per_clust = adata.obsm['other_features'].groupby('cluster').std(numeric_only=True)
    medians_per_clust_sorted = medians_per_clust['normalized_xy_dist_from_center'].sort_values(ascending=False)

    cat_index = medians_per_clust_sorted.index
    changed_clusters = {code: count for count, code in enumerate(cat_index.codes)}
    adata.obs['pheno_leiden_new'] = adata.obsm['other_features']['cluster'].map(changed_clusters)
    num_clusts = max(adata.obsm['other_features']['cluster'].values)
    adata.obs['pheno_leiden_new'] = pd.Categorical(adata.obs['pheno_leiden_new'], categories = range(num_clusts + 1), ordered=True)
    adata.obsm['other_features'] = adata.obsm['other_features'].drop('cluster', axis=1)
    adata.obsm['other_features']['cluster'] = adata.obs['pheno_leiden_new']

    fig, ax = plt.subplots(figsize=(8, 6))
    tl.umap(adata, random_state=42)
    pl.umap(adata, color='pheno_leiden_new', ax=ax, save='.svg', show=False)

    # PAGA trajectory analysis
    tl.paga(adata, groups='pheno_leiden_new')
    fig, ax = plt.subplots(figsize=(7, 6))
    pl.paga(adata, threshold=0.2, ax=ax, title='PAGA trajectory analysis', color='pheno_leiden_new', save='_thresh0.2.svg', show=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    tl.umap(adata, init_pos='paga')
    pl.umap(adata, color='pheno_leiden_new', save='_init_pos_paga.svg', ax=ax, show=False)

    reconstruct_pc_meshes(adata, pca, pca_alias, 8, [-1, -0.5, 0, 0.5, 1], output_dir)

    # Find representative cells
    finder = RepresentativeCellFinder(adata.obsm['X_pca'])
    clust_centroids = finder.find_cluster_centroids(adata.obs['pheno_leiden_new'].values)
    repr_cells = finder.find_cells_near_all_cluster_centroids(1)
    repr_cells_df = pd.DataFrame(repr_cells)
    adata.uns['repr_cells'] = repr_cells_df

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_repr_cells_umap(adata, 'pheno_leiden_new', ax)
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_variance_explained.png')
    
    viewer = view_repr_cells(adata, 'pheno_leiden_new', output_dir)
    napari.run()

    reconstruct_repr_cells_from_shcoeffs(adata, pca_alias, output_dir)

    if len(project_dirs) > 1:
        # Multiple batches, plot comparison
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_batch_umap(adata, ax)
        plt.tight_layout()

    # Polar plots of cell locations
    mpl_deep = convert_seaborn_cmap_to_mpl('deep')
    plot_clusters_polar(adata.obsm['other_features'], 'cluster', mpl_deep)
    plt.tight_layout()
    plt.savefig(output_dir / 'polar_plots_normalized.png')

    # Analyze intensity based stuff
    intensity_cols = {}
    for proj_dir in project_dirs:
        path_to_config = proj_dir / 'computefeatures/parameters.yaml'
        intensity_channels = get_intensity_cols_from_config(path_to_config)
        intensity_cols.update(intensity_channels)

    # Calculate intensity z-scores and make intensity plots
    for key, value in intensity_cols.items():
        intensity_col = value
        adata.obsm['other_features'] = plotting_tools.add_intensity_z_score_col(adata.obsm['other_features'], intensity_col, suffix=key)
        adata.obsm['other_features'][f'{key}_positive'] = np.where(adata.obsm['other_features'][f'{intensity_col}_z_score'] > 1, 1, 0)
        plot_intensity_umap(adata, key, value)
        plt.savefig(output_dir / f'{key}_intensity_umap.png', dpi=300)


    # Fix columns that prevent saving
    adata.obsp['pheno_jaccard_ig'] = adata.obsp['pheno_jaccard_ig'].tocsr()
    adata.obsm['other_features']['nm_centroid'] = adata.obsm['other_features']['nm_centroid'].astype('str')
    adata.obsm['other_features']['polairty'] = adata.obsm['other_features']['polairty'].astype('str')
    adata.obsm['other_features']['cell_centroid'] = adata.obsm['other_features']['cell_centroid'].astype('str')

    adata.write(output_dir / 'adata.h5ad')
    plt.show()


if __name__ == '__main__':
    main()
