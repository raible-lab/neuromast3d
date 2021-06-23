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
import seaborn as sns
from tifffile import imread


# Start by reading in a manifest prepared by cvapipe_analysis
path_to_manifest = Path('/home/maddy/projects/claudin_gfp_5dpf_airy_live/cvapipe_run_3/local_staging/shapemode/manifest.csv')
df_ss = pd.read_csv(path_to_manifest, index_col='CellId')
X = df_ss.values
feature_cols = df_ss.columns[-8:].tolist()


# Attempt at creating interactive plot
@magicgui(
        call_button='Create plot',
        df={'bind': df_ss},
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
    line, = ax.plot(xs, ys, 'o', picker=True, pickradius=5)
    return mpl_fig


# Function for onpick event
def onpick(event):
    ind = event.ind
    seg_path = X[ind, 1]
    print(seg_path)
    image = imread(seg_path)
    viewer.add_image(image)


viewer = napari.Viewer()
mpl_fig = create_interactive_plot(df_ss, feature_cols[0], feature_cols[1])
mpl_fig.canvas.mpl_connect('pick_event', onpick)

# Add the figure to the viewer as a FigureCanvas widget
viewer.window.add_dock_widget(FigureCanvas(mpl_fig))

# TODO: make it possible to create new plot with dropdown menu
# (this line adds a widget to do so but it doesn't function)
viewer.window.add_dock_widget(create_interactive_plot)

napari.run()
