#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Register a specified directory of 3D stacks """

import argparse
import concurrent
from functools import partial
import logging
from pathlib import Path
import sys

from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
import numpy as np
from pystackreg import StackReg


def apply_stackreg_to_multichannel_img(
        path_to_image,
        output_dir,
        alignment_channel,
        method='rigid_body',
        reference='previous'
):

    # Read in the image
    reader = AICSImage(path_to_image)
    image = reader.get_image_data('CZYX', S=0, T=0)

    # Decide which transformation method to use
    if method == 'rigid_body':
        sr = StackReg(StackReg.RIGID_BODY)

    elif method == 'translation':
        sr = StackReg(StackReg.TRANSLATION)

    elif method == 'scaled_rotation':
        sr = StackReg(StackReg.SCALED_ROTATION)

    elif method == 'affine':
        sr = StackReg(StackReg.AFFINE)

    elif method == 'bilinear':
        sr = StackReg(StackReg.BILINEAR)
        if reference == "previous":
            raise ValueError('Bilinear interpolation cannot be propagated. '
                             'Use first or mean for reference, not previous.')

    else:
        # Not sure if this is exactly right
        raise ValueError(f'Method {method} not valid.')

    # Calculate the transformation matrix from the reference channel
    tmats = sr.register_stack(
            image[alignment_channel, :, :, :],
            reference=reference
    )

    # Apply the transformation to all channels
    transformed = np.zeros_like(image)
    for ch in range(image.shape[0]):
        transformed[ch, :, :, :] = sr.transform_stack(image[ch, :, :, :])

    # Save the registered image to the output directory
    output_path = output_dir / f'{path_to_image.stem}.tiff'
    writer = ome_tiff_writer.OmeTiffWriter(output_path)
    writer.save(transformed, dimension_order='CZYX')

    # Save transformation matrices
    np.save(output_dir / f'{path_to_image.stem}_transformation_matrices.npy', tmats)


if __name__ == '__main__':

    logger = logging.getLogger(__name__)

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir', help='source dir of images to register')
    parser.add_argument('output_dir', help='desired output directory')
    parser.add_argument('extension', help='file extension, e.g. tiff')
    parser.add_argument(
            'alignment_channel',
            type=int,
            help='channel to use for calculating transformation matrices '
            'which will be applied to any other existing channels.'
    )
    parser.add_argument(
            'method',
            type=str,
            help='type of transformation to use for stackreg (e.g. rigid_body)'
    )
    parser.add_argument(
            'reference',
            type=str,
            help='reference to use for stack reg (previous, first, or mean)'
    )

    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    extension = args.extension
    alignment_ch = args.alignment_channel
    method = args.method
    reference = args.reference

    # Create output dir if not already existing
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save info into logfile
    log_file_path = output_dir / 'pystackreg.log'
    logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format='%(asctime)s %(message)s'
    )
    logger.info(sys.argv)

    # Check source dir exists
    if not source_dir.is_dir():
        print('Source dir does not exist')
        sys.exit()

    # Find all image files in source directory
    raw_files = list(source_dir.glob(f'*.{extension}'))

    # Make partial function to set constant arguments
    # Our function needs to take only one argument to work with 'map'
    apply_stackreg = partial(
            apply_stackreg_to_multichannel_img,
            output_dir=output_dir,
            alignment_channel=alignment_ch,
            method=method,
            reference=reference
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(apply_stackreg, raw_files)
