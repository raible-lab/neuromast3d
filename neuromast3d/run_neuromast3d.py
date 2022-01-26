#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import yaml

from neuromast3d.segmentation import dual_channel_annotator
from neuromast3d.alignment import nm_alignment_basic
from neuromast3d.prep_single_cells import create_fov_dataset, prep_single_cells


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


def find_steps_to_run_from_config(config):
    possible_steps = [
            'segmentation',
            'create_fov_dataset',
            'prep_single_cells',
            'alignment',
    ]
    steps_to_run = [step for step in possible_steps if config[f'{step}']['state']]
    return steps_to_run


def run_steps(steps_to_run, config):
    output_dir = Path(config['project_dir'])
    save_config(config, output_dir)
    if 'segmentation' in steps_to_run:
        dual_channel_annotator.main()
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
