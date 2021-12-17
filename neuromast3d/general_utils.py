#!usr/bin/env python3
# -*- coding: utf-8 -*-

""" Put functions used in multiple workflow steps here. """

import argparse
import sys

import yaml


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



