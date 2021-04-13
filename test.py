#!usr/bin/env python3
# -*- coding: utf-8 -*-

""" File containing tests for functions that I have written """

import unittest

import numpy as np
from skimage.draw import ellipsoid

from nm_alignment_by_pca import find_major_axis_by_pca

class FindAxisTestCase(unittest.TestCase):

    def test_find_major_axis_by_pca(self):
        """ Test that finding major axis by PCA works on an ellipsoid

        Note that the function assumes input array dims are ordered as ZYX,
        opposite the draw.ellipsoid convention.

        Also, the order of coords is flipped during the function, such that
        the columns of the resulting eigenvector matrix represent x, y, z.
        """

        ellip = ellipsoid(50, 100, 200)
        eigenvecs = find_major_axis_by_pca(ellip, threed=True)
        eigenvecs = np.absolute(eigenvecs)
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        self.assertIsNone(np.testing.assert_almost_equal(eigenvecs, expected))


if __name__ == '__main__':
    unittest.main()
