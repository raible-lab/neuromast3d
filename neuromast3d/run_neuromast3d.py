#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys

import yaml

from neuromast3d.segmentation import dual_channel_annotator
from neuromast3d.alignment import nm_alignment_basic
from neuromast3d.prep_single_cells import create_fov_dataset, prep_single_cells_for_analysis_dual_channel


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Neuromast 3d workflow script')
    parser.add_argument('config', help='path to config YAML file')
    args = parser.parse_args()
    return args


def load_config(args):
    config = yaml.load(open(args.config), Loader=yaml.Loader)
    return config


def find_steps_to_run_from_config(config):
    possible_steps = [
            'segmentation',
            'create_fov_dataset',
            # TODO: make naming scheme consistent in config file
            #'prep_single_cells',
            'alignment',
    ]
    steps_to_run = [step for step in possible_steps if config[f'{step}']['state']]
    return steps_to_run


def run_steps(steps_to_run):
    if 'segmentation' in steps_to_run:
        dual_channel_annotator.main()
    if 'create_fov_dataset' in steps_to_run:
        create_fov_dataset.main()
        prep_single_cells_for_analysis_dual_channel.main()
    if 'alignment' in steps_to_run:
        nm_alignment_basic.main()

def main():
    args = parse_cli_args()
    config = load_config(args)
    steps_to_run = find_steps_to_run_from_config(config)
    run_steps(steps_to_run)


if __name__ == '__main__':
    main()
