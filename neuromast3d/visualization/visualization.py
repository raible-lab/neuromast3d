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
df = pd.read_csv(path_to_manifest, index_col='CellId')

# Attempt at creating interactive plot
X = df.values
xs = df['MEM_PC1']
ys = df['MEM_PC2']

mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)
ax.set_title('click on points to explore the data')
line, = ax.plot(xs, ys, 'o', picker=True, pickradius=5)


# Function for onpick event
def onpick(event):
    ind = event.ind
    seg_path = X[ind, 1]
    print(seg_path)
    image = imread(seg_path)
    viewer.add_image(image)


mpl_fig.canvas.mpl_connect('pick_event', onpick)

viewer = napari.Viewer()

# Add the figure to the viewer as a FigureCanvas widget
viewer.window.add_dock_widget(FigureCanvas(mpl_fig))

napari.run()
