#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Test alignment methods prior to spherical harmonics paramterization """

import os

import numpy as np
from scipy.ndimage import center_of_mass
from skimage.measure import regionprops
from skimage.morphology import ball, binary_closing

from aicsimageio import AICSImage
from aicsshparam import shtools, shparam

project_dir = '/home/maddy/projects/claudin_gfp_5dpf_airy_live/'
test_nm = '20200617_1-O1'
test_nm_dir = f'{project_dir}/single_cell_masks/{test_nm}'

whole_seg_reader = AICSImage(
        f'{project_dir}/label_images_fixed_bg/{test_nm}_rawlabels.tiff'
        )
whole_seg_img = whole_seg_reader.get_image_data('ZYX', S=0, T=0, C=0)

whole_nm = whole_seg_img[whole_seg_img > 0].astype(np.uint8)
whole_nm = binary_closing(whole_nm, ball(5))
whole_nm_centroid = center_of_mass(whole_nm)

for root, dirs, files in os.walk(test_nm_dir):
    for dire in dirs:
        seg_filename = f'{root}/{dire}/segmentation.ome.tif'

        cell_number = dire

        seg_reader = AICSImage(seg_filename)
        seg_img = seg_reader.get_image_data('ZYX', S=0, T=0, C=0)

        ref_aligned, angle = shtools.align_image_2d(seg_img, make_unique=True)
        img_aligned = shtools.apply_image_alignment_2d(
                image=ref_aligned,
                angle=angle
            ).squeeze()

        (coeffs, grid_rec), (image_, mesh, grid_down, transform) = shparam.get_shcoeffs(
                image=img_aligned,
                lmax=16,
                sigma=2,
                compute_lcc=True,
                alignment_2d=False,
                make_unique=False
            )

        shtools.save_polydata(mesh, f'alignment_test_uniq/{cell_number}.vtk')

        mesh_rec = shtools.get_reconstruction_from_grid(grid_rec)
        shtools.save_polydata(mesh_rec, f'alignment_test_uniq/{cell_number}_rec.vtk')
