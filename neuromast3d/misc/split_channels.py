#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Short script for taking Airyscan processed images (.czi files)
Keeping only the channels up to max_channel and saving as .tiff
'''

import argparse
from pathlib import Path

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
from tqdm import tqdm


def main():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir')
    parser.add_argument('output_dir')
    parser.add_argument('extension')
    parser.add_argument('max_channel', type=int)

    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    extension = args.extension
    max_channel = args.max_channel

    assert source_dir.exists()

    # Make output directory if not already existing
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read images in source directory
    source_files = list(source_dir.glob(f'*.{extension}'))

    for fn in tqdm(source_files):
        reader = AICSImage(fn)
        image = reader.get_image_data('CZYX', S=0, T=0)
        image_subset = image[0:max_channel, :, :, :]

        output_path = output_dir / f'{fn.stem}.tiff'
        writer = ome_tiff_writer.OmeTiffWriter(output_path)
        writer.save(image_subset, dimension_order='CZYX')


if __name__ == '__main__':
    main()
