#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from aicsimageio import AICSImage
from aicsshparam import shtools, shparam

test_dir = '/home/maddy/projects/claudin_gfp_5dpf_airy_live/single_cell_masks'
test_nm_dir = f'{test_dir}/20200617_1-O1'

for root, dirs, files in os.walk(test_nm_dir):
    for dire in dirs:
        seg_filename = f'{root}/{dire}/segmentation.ome.tif'

        cell_number = dire

        seg_reader = AICSImage(seg_filename)
        seg_img = seg_reader.get_image_data('ZYX', S=0, T=0, C=0)

        ref_aligned, angle = shtools.align_image_2d(seg_img, make_unique=True)
        img_aligned = shtools.apply_image_alignment_2d(
                image = ref_aligned,
                angle = angle
            ).squeeze()

        (coeffs, grid_rec), (image_, mesh, grid_down, transform) = shparam.get_shcoeffs(
                image = img_aligned,
                lmax = 16,
                sigma = 2,
                compute_lcc = True,
                alignment_2d = False,
                make_unique = False 
            )

        shtools.save_polydata(mesh, f'alignment_test_uniq/{cell_number}.vtk')

        mesh_rec = shtools.get_reconstruction_from_grid(grid_rec)
        shtools.save_polydata(mesh_rec, f'alignment_test_uniq/{cell_number}_rec.vtk')
