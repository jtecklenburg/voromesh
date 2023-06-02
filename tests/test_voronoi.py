# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:32:44 2023

@author: jante
"""

import unittest

from voronoi import Voronoi
from numpy import array, max, min, sqrt

class VoronoiTests(unittest.TestCase):

    def setUp(self):
        self.points1 = array([[-0.5, -0.5], [-0.5, 0.0],
                              [0.5, -0.5], [0.5, 0.0],
                              [-0.5, 0.5], [0.0, -0.5],
                              [0.5, 0.5], [0.0, 0.5],
                              [0.0, 0.0]])

        self.points2 = array([[-0.5, -0.5], [0.5, -0.5],
                              [-0.5, 0.5], [0.5, 0.5],
                              [0.0, 0.0]])

        self.voro1 = Voronoi(self.points1, buffer_size=0.4)

        self.voro2 = Voronoi(self.points2, buffer_size=0.25)

    def test_buffer_size(self):
        result = self.voro1.buffer_size
        expected_result = 0.4
        self.assertEqual(result, expected_result)

        result = self.voro2.buffer_size
        expected_result = 0.25
        self.assertEqual(result, expected_result)

    def test_points(self):
        result = len(self.voro1.points)
        expected_result = len(self.points1)
        self.assertEqual(result, expected_result)

        result = len(self.voro2.points)
        expected_result = len(self.points2)
        self.assertEqual(result, expected_result)

if __name__ == "__main__":
    unittest.main()



# self.voronoi,
