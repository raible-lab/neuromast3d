#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from neuromast3d.general_utils import (parse_cli_args, load_config,
                                       find_steps_to_run_from_config, run_steps)


def main():
    args = parse_cli_args()
    config = load_config(args)
    steps_to_run = find_steps_to_run_from_config(config)
    run_steps(steps_to_run)


if __name__ == '__main__':
    main()
