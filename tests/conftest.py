#!usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
from skimage.draw import ellipsoid
import pytest
import yaml


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


@pytest.fixture
def big_sphere():
    return np.pad(ellipsoid(10, 10, 10), (5, 5))


@pytest.fixture
def small_sphere():
    return np.pad(ellipsoid(5, 5, 5), (10, 10))


@pytest.fixture
def concentric_spheres(big_sphere, small_sphere):
    # Simulates a single cell with membrane and nucleus masks
    return np.stack((big_sphere, small_sphere), axis=0)

