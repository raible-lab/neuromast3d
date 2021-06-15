#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def get_largest_cc(image):
    largest_cc = np.argmax(np.bincount(image.flat))
    return largest_cc


def switch_label_values(label_image, first, second):
    last_label = label_image.max()
    label_image = np.where(label_image == first, last_label + 1, label_image)
    label_image = np.where(label_image == second, first, label_image)
    label_image = np.where(label_image == last_label + 1, second, label_image)
    return label_image
