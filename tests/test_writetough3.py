import unittest

from voromesh import Voronoi
from voromesh.geo import layersfromsurf
from voromesh.tough3 import write_mesh
from voromesh.tough3.helper import calc_conne

import os
import numpy as np
from shapely import Polygon
from difflib import unified_diff



class TestCreateUnstructuredMesh(unittest.TestCase):

    def setUp(self):
        self.points = np.array([[0.5, 0.5], [0.5, 1.5], [0.5, 2.5],
                           [1.5, 0.5], [1.5, 1.5], [1.5, 2.5],
                           [2.5, 0.5], [2.5, 1.5], [2.5, 2.5]])

        boundary = Polygon(((0., 0.), (0., 3.), (3., 3.), (3., 0.), (0., 0.)))

        voro = Voronoi(self.points, boundary=boundary)

        mesh = voro.to_pyvista()

        self.mesh = layersfromsurf(mesh, [1])

        incon = np.full((mesh.n_cells, 4), -1.0e9)
        incon[:, 0] = 1.0
        incon[:, 1] = 2.0
        incon[:, 2] = 3.0
        incon[:, 3] = 4.0
        self.mesh.cell_data["initial_condition"] = incon

        self.mesh["initial_condition"][:] = [1.0, 2.0, 3.0, 4.0]
        self.mesh["material"] = [2, 3, 2, 3, 1, 3, 2, 3, 2]

        self.materials = {1: "CENT", 2: "EDGE", 3: "SIDE"}


    def test_cell_volumes(self):

        expected_volume = 9.0
        volume = np.sum(self.mesh["Volume"])
        self.assertEqual(volume, expected_volume)

    def test_calc_conne(self):

        distances, inner_interfaces, outer_interfaces, betax = calc_conne(self.mesh)

        expected_outer_area = 2 * 9. + 4 * 3.
        outer_area = np.sum(list(outer_interfaces.values()))
        self.assertEqual(outer_area, expected_outer_area)

        expected_inner_area = 8. + 4.
        inner_area = np.sum(list(inner_interfaces.values()))
        self.assertEqual(inner_area, expected_inner_area)

        expected_distance = 6. + 6.
        distance = np.sum(list(distances.values()))
        self.assertEqual(distance, expected_distance)

        expected_betax = 0.
        betax_ = np.sum(list(betax.values()))
        self.assertEqual(betax_, expected_betax)

    def test_write_mesh(self):

        path = os.path.dirname(os.path.realpath(__file__))
        write_mesh(path, self.mesh, self.materials)

        # check MESH
        path_expected = os.path.join(path, "ref", "MESH")
        path_actual = os.path.join(path, "MESH")

        with open(path_expected, "r") as f:
            expected_lines = f.readlines()

        with open(path_actual, "r") as f:
            actual_lines = f.readlines()

        diff = list(unified_diff(expected_lines, actual_lines))
        assert diff == [], "Unexpected file contents in MESH:\n" + "".join(diff)

        # check INCON
        path_expected = os.path.join(path, "ref", "INCON")
        path_actual = os.path.join(path, "INCON")

        with open(path_expected, "r") as f:
            expected_lines = f.readlines()

        with open(path_actual, "r") as f:
            actual_lines = f.readlines()

        diff = list(unified_diff(expected_lines, actual_lines))
        assert diff == [], "Unexpected file contents in INCON:\n" + "".join(diff)

if __name__ == "__main__":
    unittest.main()
