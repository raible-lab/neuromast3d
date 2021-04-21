#!/usr/bin/env python3
# -*- coding:utf-8 -*-

""" Add a few required columns to manifest.csv before cvapipe_analysis

Ideally, this could happen during the single cell prep stage, but since I
didn't do it then for this run, I'll do it after the fact.

The ones I need to add are structure_name and name_dict. These should be the
same for all the cells. I believe structure_name can be a placeholder like NA
since none of the internal cell structures are labeled.
"""

import pandas as pd

# Read in manifest to be updated
project_dir = '/home/maddy/projects/claudin_gfp_5dpf_airy_live/cvapipe_run_2/'
path_to_manifest = f'{project_dir}/alignment/manifest.csv'

cell_df = pd.read_csv(path_to_manifest, index_col=0)

# Add name_dict
name_dict = {
        'crop_raw': ['membrane'],
        'crop_seg': ['cell_seg']
}

num_cells = len(cell_df)
name_dict_list = [name_dict] * num_cells
cell_df['name_dict'] = name_dict_list

# Add structure_name
structure_name = 'NA'

cell_df['structure_name'] = structure_name

# Save updated manifest
cell_df.to_csv(f'{project_dir}/manifest_updated.csv')
