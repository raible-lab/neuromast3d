#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List

import yaml

from neuromast3d.segmentation import dual_channel_annotator, nucleus_segmentation
from neuromast3d.alignment import nm_alignment_basic
from neuromast3d.prep_single_cells import create_fov_dataset, prep_single_cells


POSSIBLE_STEPS = [
        'nucleus_segmentation',
        'membrane_segmentation',
        'create_fov_dataset',
        'prep_single_cells',
        'alignment'
]


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Neuromast3d workflow script')
    parser.add_argument('config', help='path to config YAML file')
    args = parser.parse_args()
    return args


def load_config(args):
    config = yaml.load(open(args.config), Loader=yaml.Loader)
    return config


def save_config(config, output_dir):
    with open(output_dir / 'config_saved.yaml', 'w') as output:
        yaml.dump(config, output)


def find_steps_to_run_from_config(config, steps=POSSIBLE_STEPS) -> List:
    steps_to_run = [step for step in steps if config[f'{step}']['state']]
    return steps_to_run


def validate_steps(steps_to_run: List, possible_steps=POSSIBLE_STEPS):
    if steps_to_run == []:
        raise ValueError('No valid steps indicated to be run, check config')
    if len(steps_to_run) > 1:
        step_indices = [possible_steps.index(step) for step in steps_to_run]
        diffs = [y - x for y, x in zip(step_indices, step_indices[1:])]
        if any(d < -1 for d in diffs):
            # A step has been skipped
            raise ValueError('Missing step(s), check config')


def run_steps(steps_to_run, config):
    output_dir = Path(config['project_dir'])
    save_config(config, output_dir)
    if 'nucleus_segmentation' in steps_to_run:
        nucleus_segmentation.execute_step(config)
    if 'membrane_segmentation' in steps_to_run:
        dual_channel_annotator.execute_step(config)
    if 'create_fov_dataset' in steps_to_run:
        create_fov_dataset.execute_step(config)
    if 'prep_single_cells' in steps_to_run:
        prep_single_cells.execute_step(config)
    if 'alignment' in steps_to_run:
        nm_alignment_basic.execute_step(config)


def main():
    args = parse_cli_args()
    config = load_config(args)
    steps_to_run = find_steps_to_run_from_config(config)
    run_steps(steps_to_run, config)


if __name__ == '__main__':
    main()
