#!usr/bin/env python3
# -*- coding: utf-8 -*-

""" Manually rotate a 3D stack around the z-axis using napari """

import argparse
import logging
from pathlib import Path
import sys
from typing import List

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
from magicgui import magicgui
import napari
from napari.types import LayerDataTuple, ImageData
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, rotate

parser = argparse.ArgumentParser(
        description='Interactive xy rotation script'
)
parser.add_argument('raw_dir')
parser.add_argument('seg_dir')
#parser.add_argument('output_dir')

args = parser.parse_args()
raw_dir = Path(args.raw_dir)
seg_dir = Path(args.seg_dir)
#output_dir = Path(args.output_dir)

#output_dir.mkdir(parents=True, exist_ok=True)

viewer = napari.Viewer()


@magicgui(call_button='Clear layers')
def clear_layers():
    if viewer.layers:
        viewer.layers.clear()
    return


# Function to explore rotation parameters
@magicgui(
        call_button='Apply rotation',
        angle={'widget_type': 'FloatSlider', 'max': 360},
        mode={'choices': ['reflect', 'constant', 'nearest', 'mirror']},
        layout='horizontal'
)
def rotate_image_interactively(
        layer: ImageData,
        angle: float = 0,
        mode='constant',
        order: int = 0,
        axes1: int = 2,
        axes2: int = 1
) -> ImageData:
    if layer is not None:
        rotated_image = rotate(
                input=layer,
                angle=angle,
                axes=(axes1, axes2),
                mode=mode,
                order=order,
                reshape=True
        )
        return rotated_image


@magicgui(
        call_button='Open image and labels',
        raw_filename={'label': 'Pick a raw file'},
        seg_filename={'label': 'Pick a seg file'}
)
def open_image(
        raw_filename: Path = raw_dir,
        seg_filename: Path = seg_dir
) -> List[LayerDataTuple]:
    reader = AICSImage(raw_filename)
    image = reader.get_image_data('ZYX', C=0, S=0, T=0)
    reader = AICSImage(seg_filename)
    labels = reader.get_image_data('ZYX', C=0, S=0, T=0)
    
    return [(image, {'name': 'raw', 'blending': 'additive'}, 'image'),
            (labels, {'name': 'labels'}, 'labels')]


viewer.window.add_dock_widget(clear_layers, area='right')
viewer.window.add_dock_widget(open_image, area='right')
viewer.window.add_dock_widget(rotate_image_interactively, area='bottom')

napari.run()
