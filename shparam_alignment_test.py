#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from aicsimageio import AICSImage
from aicsshparam import shtools, shparam

test_dir = '/home/maddy/projects/claudin_gfp_5dpf_airy_live/single_cell_masks'
test_nm_dir = f'{test_dir}/20200617_1-O1'

seg_reader = AICSImage(f'{test_nm_dir}/{0}/segmentation.ome.tif')
seg_img = seg_reader.get_image_data('ZYX', S=0, T=0, C=0)

ref_aligned, angle = shtools.align_image_2d(seg_img)
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
        make_unique = True 
    )

shtools.save_polydata(mesh, 'test_cell_0_uniq.vtk')

mesh_rec = shtools.get_reconstruction_from_grid(grid_rec)
shtools.save_polydata(mesh_rec, 'test_rec_0_uniq.vtk')
