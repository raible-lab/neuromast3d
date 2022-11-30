#/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Utility functions for data visualization and plot generation """

import ast
import logging
from pathlib import Path
from typing import Union

import matplotlib
from matplotlib import pyplot as plt
import napari
import numpy as np
import pandas as pd
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
        print('No valid alias provided. Options: NUC, MEM, or NUC_MEM')
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
