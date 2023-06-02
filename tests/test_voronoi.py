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

    def test_volume(self):
        result = len(self.voro1.volume)
        expected_result = len(self.points1)
        self.assertEqual(result, expected_result)

    def test_interface(self):
        result = 0.65
        expected_result = max(self.voro1.interface)
        self.assertEqual(result, expected_result)

        result = 0.5
        expected_result = min(self.voro1.interface)
        self.assertEqual(result, expected_result)

        result = 12
        expected_result = len(self.voro1.interface)
        self.assertEqual(result, expected_result)

        result = sqrt(2)/2
        expected_result = max(self.voro2.interface)
        self.assertEqual(result, expected_result)

        result = 0.25
        expected_result = min(self.voro2.interface)
        self.assertEqual(result, expected_result)

        result = 8
        expected_result = len(self.voro2.interface)
        self.assertEqual(result, expected_result)

    def test_neighbor(self):
        result = 12
        expected_result = len(self.voro1.neighbor)
        self.assertEqual(result, expected_result)

        result = 8
        expected_result = len(self.voro2.neighbor)
        self.assertEqual(result, expected_result)

    def test_distance(self):
        result = 12
        expected_result = len(self.voro1.distance)
        self.assertEqual(result, expected_result)

        result = 8
        expected_result = len(self.voro2.distance)
        self.assertEqual(result, expected_result)

    def test_nvec(self):
        result = 12
        expected_result = len(self.voro1.nvec)
        self.assertEqual(result, expected_result)

        result = 8
        expected_result = len(self.voro2.nvec)
        self.assertEqual(result, expected_result)

    def test_inner(self):
        result = len(self.voro1.points)
        expected_result = len(self.voro1.inner)
        self.assertEqual(result, expected_result)

        result = len(self.voro2.points)
        expected_result = len(self.voro2.inner)
        self.assertEqual(result, expected_result)

if __name__ == "__main__":
    unittest.main()



# self.voronoi,
