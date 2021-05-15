#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Segment binarized nuclei using distance transform watershed

Note: This scripts assumes a membrane channel is available as well.
"""

import argparse
from glob import glob
import os
from pathlib import Path
import sys

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
import napari
from napari.types import LabelsData, ImageData
from magicgui import magicgui
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, filters
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


@magicgui(call_button='Run DT watershed')
def dt_watershed(
        image: ImageData,
        sigma: int,
        min_distance: int
) -> LabelsData:
    distance = ndi.distance_transform_edt(image)
    distance_smoothed = filters.gaussian(
            distance,
            sigma=sigma,
            preserve_range=True
    )
    maxima = peak_local_max(
            distance_smoothed,
            min_distance=min_distance,
            exclude_border=(1, 0, 0)
    )
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(maxima.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)

    return labels


# Command line arguments
parser = argparse.ArgumentParser(
        description='DT watershed segmentation of nucleus mask'
)
parser.add_argument('raw_dir', help='directory containing raw input images')
parser.add_argument('mask_dir', help='directory containing nuclear masks')
parser.add_argument('output_dir', help='directory to save output labels')
# parser.add_argument('sigma', help='sigma to use for Gaussian blur of \
#        distance transform')
# parser.add_argument('min_distance', help='min_distance parameter for \
#        peak_local_max function')

args = parser.parse_args()

raw_dir = args.raw_dir
mask_dir = args.mask_dir
output_dir = args.output_dir
# sigma = args.sigma
# min_distance = args.min_distance

# Check raw and mask directories exist
if not os.path.isdir(raw_dir):
    print('Raw directory does not exist')
    sys.exit()

if not os.path.isdir(mask_dir):
    print('Mask directory does not exist')
    sys.exit()

# Collect all image ids in raw directory
list_of_img_ids = [fn.stem for fn in Path(raw_dir).glob('*.tiff')]

# Loop over matching raw and mask images
for img_id in list_of_img_ids:

    # Read raw image
    path = Path(f'{raw_dir}/{img_id}.tiff')
    reader = AICSImage(path)
    raw_img = reader.get_image_data('CZYX', S=0, T=0)

    # Read mask image
    path = Path(f'{mask_dir}/{img_id}_nuc_seg.tiff')
    reader = AICSImage(path)
    nuc_mask = reader.get_image_data('ZYX', C=0, S=0, T=0)

    # Split raw into membrane and nucleus channels
    # Note: Could refactor to allow passing channel index as arg
    # In case this varies
    membranes = raw_img[0, :, :, :]
    nuclei = raw_img[1, :, :, :]

    # Blur, threshold, and erode the membrane image
    mem_blurred = filters.gaussian(membranes, sigma=1, preserve_range=True)
    thresh_otsu = filters.thresholding.threshold_otsu(membranes)
    mem_otsu = mem_blurred > thresh_otsu
    mem_eroded = morphology.binary_erosion(mem_otsu)

    # Use eroded membrane mask to split touching nuclei
    nuclei_split = np.where(mem_eroded, 0, nuc_mask)

    # Inspect the 'splitting' results in the napari viewer
    viewer = napari.view_image(nuclei, colormap='gray', blending='additive')
    viewer.add_image(nuc_mask, colormap='magenta', blending='additive')
    viewer.add_image(nuclei_split, colormap='green', blending='additive')

    # Add dt watershed widget
    viewer.window.add_dock_widget(dt_watershed)
    napari.run()
