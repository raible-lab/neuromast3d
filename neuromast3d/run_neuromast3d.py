#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import pathlib
import yaml

from neuromast3d.segmentation import dual_channel_annotator
from neuromast3d.prep_single_cells import create_fov_dataset, prep_single_cells_for_analysis_dual_channel
from neuromast3d.alignment import nm_alignment_basic

'''
class WorkflowStep:

    def __init__(
            self,
            input_dir,
            output_dir,
            input_ext,
            output_ext,
            state=True
        ):
        
        self.input_dir = pathlib.Path(input_dir)
        self.output_dir = pathlib.Path(output_dir)
        self.input_ext = input_ext
        self.output_ext = output_ext
        self.state = state

        # create output directory if not already existing
        if self.state:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def step_logger(self, output_dir):
        # Not sure if this should be here or imported from elsewhere
        log_file_path = output_dir / 'logfile.log'
        logging.basicConfig(
                filename=log_file_path,
                level=logging.INFO,
                format='%(asctime)s %(message)s'
        )

    def gather_list_of_input_files(self, input_dir, input_ext):
        input_files = input_dir.input_dir('*.{input_ext}')
        return input_files
'''


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='neuromast 3d script')
    parser.add_argument('--config', help='path to the YAML config file')
    args = parser.parse_args()

    # Load config
    config = yaml.load(open(args.config), Loader=yaml.Loader)

    # Do stuff here? TODO
    # Will need to refactor downstream code...
    # Example of what this could look like:
    if config['segmentation']['state']:
        dual_channel_annotator.main()
    if config['create_fov_dataset']['state']:
        create_fov_dataset.main()
        prep_single_cells_for_analysis_dual_channel.main()
    if config['alignment']['state']:
        nm_alignment_basic.main()


if __name__ == '__main__':
    main()
