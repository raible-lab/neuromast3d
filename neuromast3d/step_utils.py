#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" General utility functions used in multiple steps. """

import logging
from pathlib import Path
import sys

from aicsimageio import AICSImage


def check_dir_exists(path_to_dir):
    if not path_to_dir.is_dir():
        print(f'{path_to_dir} does not exist')
        sys.exit()


def step_logger(step_name, output_dir):
    logger = logging.getLogger(__name__)
    log_file_path = output_dir / f'{step_name}.log'
    logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format='%(asctime)s %(message)s'
    )
    return logger


def create_step_dir(project_dir: Path, step_name: str):
    step_dir = project_dir / step_name
    step_dir.mkdir(parents=True, exist_ok=True)
    return step_dir


def read_raw_and_seg_img(path_to_raw, path_to_seg):
    reader = AICSImage(path_to_raw)
    raw_img = reader.get_image_data('CZYX', S=0, T=0)

    reader = AICSImage(path_to_seg)
    seg_img = reader.get_image_data('CZYX', S=0, T=0)
    return raw_img, seg_img

