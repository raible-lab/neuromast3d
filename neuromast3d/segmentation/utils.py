#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage as ndi
from skimage import feature, filters, segmentation


def dt_watershed(
        image: np.array,
        sigma: int,
        min_distance: int
) -> np.array:
    distance = ndi.distance_transform_edt(image)
    distance_smoothed = filters.gaussian(
            distance,
            sigma=sigma,
            preserve_range=True
    )
    maxima = feature.peak_local_max(
            distance_smoothed,
            min_distance=min_distance,
            exclude_border=(1, 0, 0)
    )
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(maxima.T)] = True
    markers, _ = ndi.label(mask)
    labels = segmentation.watershed(-distance, markers, mask=image)

    return labels
