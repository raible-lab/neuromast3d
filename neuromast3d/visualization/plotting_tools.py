#/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Utility functions for data visualization and plot generation """

import ast
import logging
from pathlib import Path
from typing import Optional, Union

from aicsimageio import AICSImage
from aicsshparam import shtools
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import napari
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import yaml
import phenograph
import seaborn as sns
from sklearn.decomposition import PCA
from tifffile import imread, imsave
import umap


logger = logging.getLogger(__name__)


def generate_pca_df_from_shcoeffs(features_df, alias='NUC_MEM'):
    shcoeffs_matrix = get_matrix_of_shcoeffs_for_pca(features_df, alias)
    axes, pca = apply_pca(shcoeffs_matrix, 0.9, alias=alias)
    axes = axes.set_index(features_df.index)
    return axes, pca


def get_matrix_of_shcoeffs_for_pca(df, alias):
    if alias == 'NUC_MEM':
        prefixes = ['MEM_shcoeffs_L', 'NUC_shcoeffs_L']
    elif alias == 'NUC':
        prefixes = ['NUC_shcoeffs_L']
    elif alias == 'MEM':
        prefixes = ['MEM_shcoeffs_L']
    else:
        logger.warning('No valid alias provided. Options: NUC, MEM, or NUC_MEM')
    features_to_use = [feat for feat in df.columns if any(word in feat for word in prefixes)]
    df_shcoeffs = df[features_to_use]
    matrix_of_features = df_shcoeffs.values.copy()
    return matrix_of_features, features_to_use


def apply_pca(matrix_of_features, n_components, alias):
    pca = PCA(n_components=0.90)
    axes = pca.fit_transform(matrix_of_features)
    p = alias
    columns = [f'{p}_PC{s}' for s in range(1, 1 + pca.n_components_)]
    axes = pd.DataFrame(axes, columns=columns)
    return axes, pca


def cluster_cells(axes, clustering_algo, resolution, seed=42, k=30):
    communities, graph, Q = phenograph.cluster(axes, k=k, clustering_algo='leiden', resolution_parameter=resolution, seed=seed)
    embedding = umap.UMAP(random_state=seed).fit_transform(axes)
    embedding_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
    embedding_df['cluster'] = communities
    g = sns.scatterplot(x='UMAP_1', y='UMAP_2', data=embedding_df, hue=embedding_df['cluster'], palette='deep')
    return embedding_df, g


def inherit_cell_labels(internal_structure, cell_seg, discard_outside=True):
    if discard_outside:
        # Make structures outside any cells part of the background
        internal_structure[cell_seg == 0] = 0
    structure_mask = np.zeros_like(cell_seg)
    structure_mask[internal_structure > 0] = 1
    structure_mask = structure_mask * cell_seg
    return structure_mask


def pick_pcs(alias, list_of_numbers):
    list_of_pcs = [f'{alias}_PC{num}' for num in list_of_numbers]
    return list_of_pcs


def add_intensity_z_score_col(df_clustered, col_name: str, suffix: str):
    df_sost_by_fov = pd.DataFrame()
    df_sost_by_fov[f'mean_{suffix}'] = df_clustered.groupby(['fov_id']).mean()[col_name]
    df_sost_by_fov[f'sd_{suffix}'] = df_clustered.groupby(['fov_id']).std()[col_name]
    df_clustered_copy = df_clustered.copy()
    df_clustered_copy = df_clustered_copy.join(df_sost_by_fov, on='fov_id')
    df_clustered_copy[f'{col_name}_z_score'] = (df_clustered_copy[col_name] - df_clustered_copy[f'mean_{suffix}']) / df_clustered_copy[f'sd_{suffix}']
    return df_clustered_copy


def calculate_intensity_z_scores(df, col_name: str, suffix: str):
    df_by_fov = pd.DataFrame()
    df_by_fov[f'mean_{suffix}'] = df.groupby(['fov_id']).mean()[col_name]
    df_by_fov[f'sd_{suffix}'] = df.groupby(['fov_id']).std()[col_name]
    z_scores = (df_by_fov[col_name] - df_by_fov[f'mean_{suffix}']) / df_by_fov[f'sd_{suffix}']
    return z_scores


def get_features_data(local_staging_dir):
    cf_path = local_staging_dir / 'computefeatures/manifest.csv'
    df = pd.read_csv(cf_path, index_col='CellId')
    # Delete rows with NaNs - kinda hacky
    df = df.dropna(axis=0, how='any', subset=['NUC_connectivity_cc'])
    return df


def subcluster(df, cluster, cluster_col_name='cluster', alias='NUC_MEM'):
    # Subclustering, with or without recomputing PCs (idk which is better)
    # I hope it's right to do it this way, it's really hard for me to tell what people usually do
    # because it's buried in R source code...
    # TBH there's probably not enough cells for this anyway
    # Subset by cluster and rerun PCA
    df_subset = df.loc[df[f'{cluster_col_name}'] == cluster]
    axes_subset, pca_subset = generate_pca_df_from_shcoeffs(df_subset, alias=alias)
    embedding_df_subset, graph = cluster_cells(axes_subset, k=30, clustering_algo='leiden', resolution=1)
    return embedding_df_subset, graph


def combine_features_datasets(project_dirs: list):
    df = pd.DataFrame()
    for count, directory in enumerate(project_dirs):
        features_data = get_features_data(directory)
        features_data['batch'] = count + 1
        df = pd.concat([df, features_data])
    return df


def fit_pca_to_subset(df, subset_col: str, subset_vals: list, alias: str = 'NUC_MEM', n_comps: Union[float, int] = 0.90):
    df_subset = df.loc[df[subset_col].isin(subset_vals)]
    shcoeffs_matrix, _  = get_matrix_of_shcoeffs_for_pca(df_subset, alias)

    # fit pca to subset
    pca = PCA(n_components=n_comps)
    pca.fit(shcoeffs_matrix)
    return pca


def apply_pca_transform(df, pca, alias: str = 'NUC_MEM'):
    shcoeffs_matrix, _ = get_matrix_of_shcoeffs_for_pca(df, alias)
    axes = pca.transform(shcoeffs_matrix)
    p = alias
    columns = [f'{p}_PC{s}' for s in range(1, 1 + pca.n_components_)]
    axes = pd.DataFrame(axes, columns=columns)
    axes = axes.set_index(df.index)
    return axes


def drop_manually_curated_cells(cell_df, curated_df):
    curated_df = curated_df.copy()
    curated_df['cells_to_exclude'] = curated_df['cells_to_exclude'].apply(ast.literal_eval)
    curated_df = curated_df.explode('cells_to_exclude')
    cellids_to_exclude = curated_df['fov_id'] + '_' + curated_df['cells_to_exclude'].astype(str)
    cellids_to_exclude = list(cellids_to_exclude.values)
    cell_df_cleaned = cell_df.drop(cellids_to_exclude, axis=0, errors='ignore')
    num_rows_removed = cell_df.shape[0] - cell_df_cleaned.shape[0]
    print(f'{num_rows_removed} cells removed')
    return cell_df_cleaned


def get_genotypes_from_cellids(cell_df) -> list:
    genotypes = []
    for row in cell_df.itertuples(index=True):
        if 'mut' in row[0].split('_'):
            genotypes.append('mut')
        elif 'wt' in row[0].split('_'):
            genotypes.append('het')
        else:
            genotypes.append('wt')
    return genotypes


def get_list_of_basic_features(aliases: list = ['NUC', 'MEM']):
    list_of_features = []
    for alias in aliases:
        list_of_features.extend([
            f'{alias}_shape_volume_lcc',
            f'{alias}_position_depth',
            f'{alias}_position_height',
            f'{alias}_position_width',
            f'{alias}_roundness_surface_area'
        ])
    return list_of_features


def get_list_of_intensity_features(alias: str):
    list_of_features = [
        f'{alias}_intensity_mean_lcc',
        f'{alias}_intensity_std_lcc',
        f'{alias}_intensity_1pct_lcc',
        f'{alias}_intensity_99pct_lcc',
        f'{alias}_intensity_min_lcc',
        f'{alias}_intensity_max_lcc'
    ]
    return list_of_features

# TODO: there is also rotation angle(s) and centroid position
# will have to think about adding those, + any other engineered features

def color_code_fov_images_by_feature(cell_dataset, feature_col, save):
    # TODO: this is bugged if the feature you are coloring by uses integers...
    fov_dataset = cell_dataset.drop_duplicates(subset='fov_id', keep='first')
    for fov in fov_dataset.itertuples():
        raw_img = imread(fov.fov_path)
        print(raw_img.shape)

        seg_img = imread(fov.fov_seg_path)
        print(seg_img.shape)

        seg_img_nuc = seg_img[0, :, :, :]
        seg_img_mem = seg_img[1, :, :, :]
        fov_cell_df_subset = cell_dataset.loc[cell_dataset['fov_id'] == fov.fov_id]
        cell_label_list = fov_cell_df_subset['label'].to_list()

        # Remove labels not in the list (e.g. that didn't pass QC)
        mask = np.zeros_like(seg_img_mem)
        for label in cell_label_list:
            mask[seg_img_mem == label] = 1
        seg_img_mem_cleaned = np.where(mask == 1, seg_img_mem, 0)

        # Color cells remaining by feature
        seg_mem_clust = seg_img_mem_cleaned.copy()
        for row in fov_cell_df_subset.itertuples():
            seg_mem_clust = np.where(seg_mem_clust == row.label, row._asdict()[feature_col], seg_mem_clust)

        # Propagate cluster color to nuclear channel
        nuc_mask = np.zeros_like(seg_img_nuc)
        nuc_mask[seg_img_nuc > 0] = 1
        seg_nuc_clust = inherit_cell_labels(seg_img_nuc, seg_mem_clust)
        seg_nuc_clust = np.where(mask * nuc_mask == 1, seg_nuc_clust, 0)

        # Inspect in napari
        viewer = napari.Viewer()
        viewer.add_image(raw_img)
        viewer.add_image(seg_mem_clust)
        viewer.add_image(seg_nuc_clust)
        napari.run()

        if save:
            seg_mem_clust = np.expand_dims(seg_mem_clust, axis=0)
            seg_nuc_clust = np.expand_dims(seg_nuc_clust, axis=0)
            merged_image = np.concatenate([seg_nuc_clust, seg_mem_clust], axis=0)
            save_dir = Path(project_dirs[0] / f'fovs_colored_by{feature}')
            save_dir.mkdir(parents=True, exist_ok=True)
            imsave(save_dir / f'{fov.fov_id}.tiff', merged_image)


def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


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
    adata,
    alias: str,
    n_comps: Union[float, int],
    batches: Optional[list] = None,
    use_highly_variable: Optional[bool] = None,
):
    # Fit PCA to subset of data, apply transform to rest, then add to adata object
    # Modified from original scanpy function
    if batches is not None:
        subset = adata[adata.obs['batch'].isin(batches)]
        df = subset.to_df()

    else:
        df = adata.to_df()

    shcoeffs, _ = get_matrix_of_shcoeffs_for_pca(df, alias)
    pca_ = PCA(n_components=n_comps)
    pca_.fit(shcoeffs)
    axes = apply_pca_transform(adata.to_df(), pca_, alias=alias)

    X_pca = axes.values
    adata.obsm['X_pca'] = X_pca
    adata.uns['pca'] = {}
    adata.uns['pca']['params'] = {
        'zero_center': True,
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


def calculate_cell_positions(df):
    df_new = pd.DataFrame() 
    df_new['nm_centroid'] = df['nm_centroid'].apply(ast.literal_eval)
    df_new['cell_centroid'] = df['centroid'].apply(ast.literal_eval)
    dist = df_new['cell_centroid'].apply(np.array) - df_new['nm_centroid'].apply(np.array)

    # Sign of y is flipped to convert from rc coords to xy coords
    df_new['z_dist'], df_new['y_dist'], df_new['x_dist'] = zip(*dist)
    df_new['y_dist'] = -df_new['y_dist']

    xy_dist = df_new[['y_dist', 'x_dist']].values.tolist()
    df_new['raw_xy_dist_from_center'] = [np.linalg.norm(row) for row in xy_dist]

    # Convert rotation angle to radians for plottings
    df_new['rotation_angle'] = df['rotation_angle']
    df_new['rotation_angle_in_rads'] = df_new['rotation_angle']*np.pi/180

    # DV should be counterclockwise rotated by 90 degrees
    df_new['corrected_rotation_angle'] = df_new['rotation_angle']
    df_new['polairty'] = df['polairty']
    df_new.loc[df_new['polairty'] == 'DV', 'corrected_rotation_angle'] = df_new['rotation_angle'] + 90
    df_new['corrected_rotation_angle_in_rads'] = df_new['corrected_rotation_angle']*np.pi/180

    # Normalize to centroid of cell furthest from the neuromast center
    df_new['fov_id'] = df['fov_id']
    groups = df_new.groupby(['fov_id'])
    max_vals = groups.transform('max')
    df_new['normalized_xy_dist_from_center'] = df_new['raw_xy_dist_from_center'] / max_vals['raw_xy_dist_from_center']
    return df_new


def convert_seaborn_cmap_to_mpl(sns_cmap):
    mpl_cmap = ListedColormap(sns.color_palette(sns_cmap).as_hex())
    return mpl_cmap


def plot_clusters_polar(df, col_name: str, cmap):
    clust_groups = df.groupby(col_name)
    fig, axs = plt.subplots(ncols=len(clust_groups), figsize=(12, 2), subplot_kw={'projection': 'polar'}, sharey=True)
    for name, group in clust_groups:
        # this assumes clusters are integers from 0 to n clusters
        axs[name].plot(
                group['corrected_rotation_angle_in_rads'].values,
                group['normalized_xy_dist_from_center'].values,
                '.',
                color=cmap(name),
                markersize=2
            )
    return axs


def plot_pca_var_explained(adata, ax, save_path: Union[str, Path, None] = None):
    ax[0].plot(adata.uns['pca']['variance_ratio'], color='darkgray', lw=2, marker='o')
    ax[0].set_xlabel('Number of principal components')
    ax[0].set_ylabel('Variance explained')
    ax[1].plot(np.cumsum(adata.uns['pca']['variance_ratio']), color='darkgray', lw=2, marker='o')
    ax[1].set_xlabel('Number of PCs')
    ax[1].set_ylabel('Cumulative variance explained')
    ax[0].set_xticks(np.arange(0, adata.obsm['X_pca'].shape[1]+1, 10))
    ax[1].set_xticks(np.arange(0, adata.obsm['X_pca'].shape[1]+1, 10))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    return ax


def plot_umap_from_adata(adata, ax=None, save_path: Union[str, Path, None] = None, **sns_kwargs):
    """
    Basic function for plotting UMAPs from processed AnnData objects.

    Parameters
    ----------
    adata: AnnData object. Must have been processed with scanpy.pl.umap.

    ax : matplotlib axes object, such as that created by plt.subplots().
        If None (the default), will use plt.gca() to provide the axes object.

    save_path: The absolute filepath where the plot will be saved. 
        If None (the default), the plot is not saved.

    **sns_kwargs: Keyword arguments to pass to sns.scatterplot.

    Returns
    -------
    ax: matplotlib axes object containing the plot.

    """
    if ax is None:
        ax = plt.gca()

    UMAP_1 = adata.obsm['X_umap'][:, 0]
    UMAP_2 = adata.obsm['X_umap'][:, 1]

    g = sns.scatterplot(x=UMAP_1, y=UMAP_2, ax=ax, **sns_kwargs)
    g.set_xlabel('UMAP 1')
    g.set_ylabel('UMAP 2')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    return ax


def plot_repr_cells_umap(adata, ax=None, save_path: Union[str, Path, None] = None, **sns_kwargs):
    if ax is None:
        ax = plt.gca()

    UMAP_1 = adata.obsm['X_umap'][:, 0]
    UMAP_2 = adata.obsm['X_umap'][:, 1]

    g = sns.scatterplot(x=UMAP_1, y=UMAP_2, **sns_kwargs)
    g.set_xlabel('UMAP 1')
    g.set_ylabel('UMAP 2')

    repr_cells_df = adata.uns['repr_cells']
    plt.scatter(UMAP_1[repr_cells_df['inds']], UMAP_2[repr_cells_df['inds']], s=70, color='black', edgecolors='white')
    plt.legend(title='cluster', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    return ax


def view_repr_cells(adata, col_name, output_dir = None):
    custom_colors = ['#000000'] + list(adata.uns[f'{col_name}_colors'])
    color_mapping = {cl: col for cl, col in enumerate(custom_colors)}
    print(color_mapping)

    viewer = napari.Viewer()
    for count, ind in enumerate(adata.uns['repr_cells']['inds']):
        path_to_seg = adata.obsm['other_features'].iloc[ind].crop_seg
        seg_img = imread(path_to_seg)
        img_ch0 = seg_img[0, :, :, :]
        img_ch1 = seg_img[1, :, :, :]

        cluster = adata.obs[col_name].iloc[ind]
        img_ch0 = np.where(img_ch0 > 0, cluster + 1, 0)
        img_ch1 = np.where(img_ch1 > 0, cluster + 1, 0)

        viewer.add_labels(img_ch0, name=f'{cluster}_ch0', blending='additive', color=color_mapping)
        viewer.add_labels(img_ch1, name=f'{cluster}_ch1', blending='additive', color=color_mapping)
 
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
    plt_args = {'edgecolor': None, 's': 70}
    fig, ax = plt.subplots(figsize=(20, 7), ncols=2)
    subset = adata[adata.obsm['other_features'][ch_name].notna()]
    # Sort so that cells with higher vals on top in the plot
    order = np.argsort(subset.obs[int_col])
    subset = subset[order]
    UMAP_1 = subset.obsm['X_umap'][:, 0]
    UMAP_2 = subset.obsm['X_umap'][:, 1]
    g = sns.scatterplot(data=subset.obs, x=UMAP_1, y=UMAP_2,
                        hue=int_col, palette='viridis', alpha=0.8, **plt_args, ax=ax[0])
    g2 = sns.scatterplot(data=subset.obs, x=UMAP_1, y=UMAP_2,
                         hue=f'{ch_name}_positive', palette=['silver', 'fuchsia'], alpha=0.8, **plt_args, ax=ax[1])
    ax[0].legend(title='intensity z-score', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    ax[1].legend(title=f'{ch_name} positive', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    g.set_xlabel('UMAP 1')
    g.set_ylabel('UMAP 2')
    g2.set_xlabel('UMAP 1')
    g2.set_ylabel('UMAP 2')
    plt.tight_layout()


def classify_rec_error(adata, project_dirs, alias):
    rec_errors = pd.DataFrame()
    for proj_dir in project_dirs:
        rec_error = pd.read_csv(proj_dir / f'rec_errors_{alias}.csv', index_col='CellId')
        rec_error = rec_error[rec_error.index.isin(adata.obs_names)]
        rec_errors = pd.concat([rec_errors, rec_error])

    adata.obsm['rec_error'] = rec_errors
    # Consider moving this to the error analysis script
    gm_model = GaussianMixture(n_components=2, covariance_type='tied')
    gm_model.fit(rec_errors['max_hd'].values.reshape(-1, 1))
    adata.obsm['rec_error']['gmm_classes'] = gm_model.predict(rec_errors['max_hd'].values.reshape(-1, 1))
    return adata


def reorder_clusters(df, by, clust_col, measure, ascending):
    if measure == 'mean':
        clust_vals = df.groupby(clust_col).mean(numeric_only=True)
    if measure == 'median':
        clust_vals = df.groupby(clust_col).median(numeric_only=True)
    
    clust_vals_sorted = clust_vals[by].sort_values(ascending=ascending)
    cat_index = clust_vals_sorted.index
    cluster_mapping = {code: count for count, code in enumerate(cat_index.codes)}
    clust_col_remapped = df[clust_col].map(cluster_mapping)
    num_clusts = max(cat_index)
    clust_categorical = pd.Categorical(clust_col_remapped, range(num_clusts + 1), ordered=True)
    return clust_categorical
