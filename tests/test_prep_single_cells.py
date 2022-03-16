#!usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from neuromast3d.prep_single_cells.prep_single_cells import execute_step, inherit_labels, create_cropping_roi


def test_execute_step(prep_single_cells_config, output_dir):
    execute_step(prep_single_cells_config)

    pred_output_dir = output_dir / 'prep_single_cells'
    assert pred_output_dir.is_dir()

    pred_output_csv = pred_output_dir / 'cell_manifest.csv'
    assert pred_output_csv.exists()


def test_inherit_labels(small_sphere, big_sphere):
    parent = big_sphere * 42
    child = small_sphere * 7
    child_expected = small_sphere * 42
    child_labeled = inherit_labels(child, parent)
    np.testing.assert_allclose(child_labeled, child_expected)


def test_create_cropping_roi(small_sphere):
    expected = [1,  33, 0, 33, 0, 33]
    actual = create_cropping_roi(small_sphere)
    np.testing.assert_allclose(actual, expected)