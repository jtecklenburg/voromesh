# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 16:00:01 2023

@author: jante
"""

import unittest
import vtk

from voronoi.io import create_unstructured_mesh


class TestCreateUnstructuredMesh(unittest.TestCase):
    def test_create_unstructured_mesh(self):
        # Define input data
        points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        connectivities = [[0, 1, 2, 3]]

        # Call function to create VTK unstructured mesh
        mesh = create_unstructured_mesh(points, connectivities)

        # Assert that the mesh has one cell of type vtkPolygon
        self.assertEqual(mesh.GetNumberOfCells(), 1)
        cell = mesh.GetCell(0)
        self.assertIsInstance(cell, vtk.vtkPolygon)

        # Assert that the mesh has four vertices at the expected positions
        self.assertEqual(mesh.GetNumberOfPoints(), 4)
        for i, expected_point in enumerate(points):
            point = mesh.GetPoint(i)
            self.assertEqual(point, expected_point)

        # Assert that the cell has the expected vertex indices
        expected_connectivity = connectivities[0]
        n_points = cell.GetNumberOfPoints()
        self.assertEqual(n_points, len(expected_connectivity))
        for j in range(n_points):
            point_id = cell.GetPointId(j)
            self.assertEqual(point_id, expected_connectivity[j])
