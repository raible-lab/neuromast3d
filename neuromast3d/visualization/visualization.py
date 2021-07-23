#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Interactive visualization of neuromast cells after analysis

The idea of this script is to be able to generate interactive plots 
using matplotlib and display them in napari, such that if you click on a
point in the plot, it will open the image of that cell in 3D.
"""

from pathlib import Path

import numpy as np
import napari
from magicgui import magicgui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
import pandas as pd
import phenograph
import seaborn as sns
from sklearn.decomposition import PCA
from tifffile import imread
import umap


def get_matrix_of_shcoeffs_for_pca(df):
    prefixes = 'MEM_shcoeffs_L', 'NUC_shcoeffs_L'
    features_to_use = [f for f in df.columns if any(w in f for w in prefixes)]
    df_shcoeffs = df[features_to_use]
    matrix_of_features = df_shcoeffs.values.copy()
    return matrix_of_features


def apply_pca(matrix_of_features, n_components, prefix):
    pca = PCA(n_components=n_components)
    axes = pca.fit_transform(matrix_of_features)
    columns = [f'{prefix}_PC{c}' for c in range(1, 1 + axes.shape[1])]
    axes = pd.DataFrame(axes, columns=columns)
    return axes


def cluster_cells(axes, clustering_algo='leiden'):
    communities, graph, Q = phenograph.cluster(axes, clustering_algo=clustering_algo)
    embedding = umap.UMAP().fit_transform(axes)
    embedding_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
    embedding_df['cluster'] = communities
    g = sns.scatterplot(data=embedding_df, x='UMAP_1', y='UMAP_2',
                        hue='cluster', palette='deep')
    return embedding_df, g


# TODO: refactor to use functions
# Start by reading in a manifest prepared by cvapipe_analysis
path_to_manifest = Path('/home/maddy/projects/claudin_gfp_5dpf_airy_live/cvapipe_run_4/local_staging/computefeatures/manifest.csv')
df_cf = pd.read_csv(path_to_manifest, index_col='CellId')

# Apply PCA to shcoeffs
shcoeffs_matrix = get_matrix_of_shcoeffs_for_pca(df_cf)
axes = apply_pca(shcoeffs_matrix, 0.9, 'MEM')
axes = axes.set_index(df_cf.index)

# Cluster, embed in UMAP space, and plot
embedding_df, graph = cluster_cells(axes, clustering_algo='leiden')
embedding_df = embedding_df.set_index(df_cf.index)

# Merge to other data frames for maximum info
path_to_manifest_a = Path('/home/maddy/projects/claudin_gfp_5dpf_airy_live/cvapipe_run_4/alignment/manifest.csv')
df_align = pd.read_csv(path_to_manifest_a, index_col='CellId')
df_clustered = pd.concat([df_align, axes, embedding_df], axis=1)

#df_ss = pd.merge(df_cf, embedding_df, left_index=True, right_index=True)
feature_cols = df_clustered.columns[-11:].tolist()


# Attempt at creating interactive plot
@magicgui(
        call_button='Create plot',
        df={'bind': df_clustered},
        xcol={'choices': feature_cols},
        ycol={'choices': feature_cols},
        result_widget=True
)
def create_interactive_plot(df, xcol, ycol):
    xs = df[f'{xcol}']
    ys = df[f'{ycol}']
    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)
    ax.set_title('click on points to explore the data')
    line = sns.scatterplot(data=df, x=xs, y=ys, hue='cluster', picker=True, pickradius=5, palette='deep')
    #line, = ax.plot(xs, ys, 'o', picker=True, pickradius=5)
    return mpl_fig


# Function for onpick event
def onpick(event):
    ind = event.ind
    # seg_path = X[ind, 1]
    picked_cell_label = df_clustered['label'].iloc[ind].values
    fov_seg_path = df_clustered['fov_seg_path'].iloc[ind].values
    single_cell_path = df_clustered['crop_seg_pre_alignment'].iloc[ind].values
    single_cell = imread(single_cell_path)
    aligned_single_cell_path = df_clustered['crop_seg'].iloc[ind].values
    aligned_single_cell = imread(aligned_single_cell_path)
    fov_image = imread(fov_seg_path)
    selected_cell = np.where(fov_image == picked_cell_label, picked_cell_label, 0)
    angle = df_clustered['rotation_angle'].iloc[ind].values
    print(fov_seg_path, picked_cell_label, angle)
    viewer.add_labels(fov_image)
    viewer.add_labels(selected_cell)
    viewer.add_image(single_cell)
    viewer.add_image(aligned_single_cell)


viewer = napari.Viewer()
mpl_fig = create_interactive_plot(df_clustered, feature_cols[-3], feature_cols[-2])
mpl_fig.canvas.mpl_connect('pick_event', onpick)

# Add the figure to the viewer as a FigureCanvas widget
viewer.window.add_dock_widget(FigureCanvas(mpl_fig))

# TODO: make it possible to create new plot with dropdown menu
# (this line adds a widget to do so but it doesn't function)
viewer.window.add_dock_widget(create_interactive_plot)

napari.run()
