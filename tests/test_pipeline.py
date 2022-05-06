#!usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
import yaml

from neuromast3d.alignment import nm_alignment_basic
from neuromast3d.prep_single_cells import create_fov_dataset
from neuromast3d.prep_single_cells import prep_single_cells


@pytest.fixture
def input_dir():
    return Path('./tests/resources/test_files').resolve()


@pytest.fixture(scope='class')
def output_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('output')


@pytest.fixture
def config_path():
    return Path('./tests/resources/config.yaml').resolve()


@pytest.fixture
def base_config(config_path, input_dir, output_dir):
    config = yaml.load(open(config_path), Loader=yaml.Loader)
    config['project_dir'] = output_dir
    config['raw_dir'] = input_dir / 'stack_aligned'
    return config


@pytest.mark.slow
class TestPipeline:

    @pytest.fixture
    def create_fov_dataset_config(self, base_config, input_dir):
        config = base_config
        config['create_fov_dataset']['state'] = True
        config['create_fov_dataset']['original_dir'] = input_dir / 'original'
        config['create_fov_dataset']['seg_dir'] = input_dir / 'segmentations'
        return config


    def test_create_fov_dataset(self, create_fov_dataset_config, output_dir):
        create_fov_dataset.execute_step(create_fov_dataset_config)
        # Check expected output files created
        pred_output_dir = output_dir
        assert pred_output_dir.is_dir()

        pred_output_csv = pred_output_dir / 'fov_dataset.csv'
        assert pred_output_csv.exists()

    @pytest.fixture
    def prep_single_cells_config(self, base_config):
        config = base_config
        config['prep_single_cells']['state'] = True
        return config


    def test_prep_single_cells(self, prep_single_cells_config, output_dir):
        prep_single_cells.execute_step(prep_single_cells_config)

        pred_output_dir = output_dir / 'prep_single_cells'
        assert pred_output_dir.is_dir()

        pred_output_csv = pred_output_dir / 'cell_manifest.csv'
        assert pred_output_csv.exists()

    
    @pytest.fixture(params=['', 'xy_xz'])
    def alignment_config(self, base_config, request):
        config = base_config
        config['alignment']['state'] = True
        config['alignment']['rot_ch_index'] = 1
        config['alignment']['make_unique'] = True
        config['alignment']['mode'] = f'{request.param}'
        return config


    def test_nm_alignment_basic(self, alignment_config, output_dir):
        nm_alignment_basic.execute_step(alignment_config)

        pred_output_dir = output_dir / 'alignment'
        assert pred_output_dir.is_dir()

        pred_output_csv = output_dir / 'alignment'
        assert pred_output_csv.exists()
