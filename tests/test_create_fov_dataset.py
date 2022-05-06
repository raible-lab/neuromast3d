#!usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path

import pytest

from neuromast3d.prep_single_cells.create_fov_dataset import get_channel_ids, create_name_dict, execute_step


def test_create_name_dict(base_config):
    raw_channel_ids, seg_channel_ids = get_channel_ids(base_config['channels'])
    name_dict = create_name_dict(raw_channel_ids, seg_channel_ids)
    assert name_dict == {
        'crop_raw': ['nucleus', 'membrane'],
        'crop_seg': ['nuc_seg', 'cell_seg']
    }
