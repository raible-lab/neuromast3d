#!usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path

import pytest

from neuromast3d.prep_single_cells.create_fov_dataset import get_channel_ids, create_name_dict, execute_step


def test_create_name_dict(create_fov_dataset_config):
    raw_channel_ids, seg_channel_ids = get_channel_ids(create_fov_dataset_config['channels'])
    name_dict = create_name_dict(raw_channel_ids, seg_channel_ids)
    assert name_dict == {
        'crop_raw': ['nucleus', 'membrane'],
        'crop_seg': ['nuc_seg', 'cell_seg']
    }


def test_execute_step(create_fov_dataset_config, output_dir):
    execute_step(create_fov_dataset_config)
    # Check expected output files created
    pred_output_dir = output_dir
    assert pred_output_dir.is_dir()

    pred_output_csv = pred_output_dir / 'fov_dataset.csv'
    assert pred_output_csv.exists()