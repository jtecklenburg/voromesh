# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:49:20 2022

@author: jante
"""

import matplotlib.pyplot as plt
from .helper import convex_hull, voronoi_bound
from .io import to_vtk, to_pyvista
from numpy import array, shape
from scipy.spatial import Delaunay
from shapely.geometry.polygon import Polygon


class Voronoi:
    def __init__(self, points, buffer_size=1, boundary=None):
        """
        Bounded 2D Voronoi decomposition for given points.
        The boundary is either calculated from the buffered convex hull of the
        given points with the given buffer_size (default) or passed directly.

        Parameters
        ----------
        points : numpy.array of shape (npoints, ndim=2)
            Coordinates of points to construct a Voronoi diagram from.
        buffer_size : float, Optional
            Buffer for convex hull >= 0.
        boundary : shapely.geometry.polygon.Polygon, Optional
            Boundary for Voronoi decomposition.

        Returns
        -------
        None.

        """

        n = shape(points)

        if not n[1] == 2:
            raise TypeError("Points needs to be an array of shape Nx2, but is "
                            + str(n[0]) + "x" + str(n[1]) + "!")

        if buffer_size < 0:
            raise ValueError("Buffer_size needs to be >= 0!")

        if boundary is None:
            self.buffer_size = buffer_size
            boundary = convex_hull(points, buffer_size)
        elif type(boundary) is Polygon:
            self.buffer_size = None
        else:
            self.buffer_size = None
            raise Warning("Boundary is not of type"
                          + "shapely.geometry.polygon.Polygon.\n"
                          + "This can lead to unexpected results!")

        self.voronoi, self.points = voronoi_bound(points, boundary)

    def plot(self, center=False, tri=False):
        """
        plots the Voronoi diagramm.

        Parameters
        ----------
        center : Bool, optional
            Add center points to plot. The default is False.
        tri : Bool, optional
            Add Dilaunay triangulation to plot. The default is False.

        Returns
        -------
        None.

        """
        if tri:
            tri = Delaunay(self.points)
            plt.triplot(self.points[:, 0], self.points[:, 1], tri.simplices)

        if center:
            plt.plot(self.points[:, 0], self.points[:, 1], 'o')

        for r in self.voronoi.geoms:
            plt.fill(*zip(*array(list(
                zip(r.boundary.coords.xy[0][:-1],
                    r.boundary.coords.xy[1][:-1])))),
                alpha=0.4)

        plt.axis('equal')
        plt.show()

    def to_vtk(self, decimalplace=8, dim="Z"):
        """
        Converts the Voronoi diagram to a vtk unstructured mesh.

        Returns
        -------
        None.

        """
        return to_vtk(self.voronoi, decimalplace, dim)

    def to_pyvista(self, decimalplace=8, dim="Z"):
        """
        Converts the Voronoi diagram to a pyvista unstructured mesh.

        Returns
        -------
        None.

        """
        return to_pyvista(self.voronoi, decimalplace, dim)

# def test():

#     # points = array([[-30.0, 30.370371], [-27.777777, 35.925926], [-34.444443, 58.51852], [-2.9629631, 57.777779], [-17.777779, 75.185181], [-29.25926, 58.148151], [-11.111112, 33.703705], [-11.481482, 40.0], [-27.037037, 40.0], [-7.7777777, 94.444443], [-2.2222223, 122.22222], [-20.370371, 106.66667], [1.1111112, 125.18518], [-6.2962961, 128.88889], [6.666667, 133.7037], [11.851852, 136.2963], [8.5185184, 140.74074], [20.370371, 92.962959], [17.777779, 114.81482], [12.962962, 97.037041], [13.333334, 127.77778], [22.592592, 120.37037], [16.296295, 127.77778], [11.851852, 50.740742], [20.370371, 54.814816], [19.25926, 47.40741], [32.59259, 122.96296], [20.74074, 130.0], [24.814816, 84.814819], [26.296295, 91.111107], [56.296295, 131.48149], [60.0, 141.85185], [32.222221, 136.66667], [53.703705, 147.03703], [87.40741, 196.2963], [34.074074, 159.62964], [34.444443, -2.5925925], [36.666668, -1.8518518], [34.074074, -7.4074073], [35.555557, -18.888889], [76.666664, -39.629627], [35.185184, -37.777779], [25.185184, 14.074074], [42.962959, 32.962963], [35.925926, 9.2592592], [52.222221, 77.777779], [57.777779, 92.222221], [47.037041, 92.59259], [82.222221, 54.074074], [48.888889, 24.444445], [35.925926, 47.777779], [50.740742, 69.259254], [51.111111, 51.851849], [56.666664, -12.222222], [117.40741, -4.4444447], [59.629631, -5.9259262], [66.666664, 134.07408], [91.481483, 127.40741], [66.666664, 141.48149], [53.703705, 4.0740738], [85.185181, 11.851852], [69.629631, 0.37037039], [68.518517, 99.259262], [75.185181, 100.0], [70.370369, 113.7037], [74.444443, 82.59259], [82.222221, 93.703697], [72.222221, 84.444443], [77.777779, 167.03703], [88.888893, 168.88889], [73.703705, 178.88889], [87.037041, 123.7037], [78.518517, 97.037041], [95.555557, 52.962959], [85.555557, 57.037041], [90.370369, 23.333332], [100.0, 28.51852], [88.888893, 37.037037], [87.037041, -42.962959], [89.259262, -24.814816], [93.333328, 7.4074073], [98.518517, 5.185185], [92.59259, 1.4814816], [85.925919, 153.7037], [95.555557, 154.44444], [92.962959, 150.0], [97.037041, 95.925919], [106.66667, 115.55556], [92.962959, 114.81482], [108.88889, 56.296295], [97.777779, 50.740742], [94.074081, 89.259262], [96.666672, 91.851852], [102.22222, 77.777779], [107.40741, 40.370369], [105.92592, 29.629629], [105.55556, -46.296295], [118.51852, -47.777779], [112.22222, -43.333336], [112.59259, 25.185184], [115.92592, 27.777777], [112.59259, 31.851852], [107.03704, -36.666668], [118.88889, -32.59259], [114.07408, -25.555555], [115.92592, 85.185181], [105.92592, 18.888889], [121.11111, 14.444445], [129.25926, -28.51852], [127.03704, -18.518518], [139.25926, -12.222222], [141.48149, 3.7037036], [137.03703, -4.814815], [153.7037, -26.666668], [-2.2222223, 5.5555558], [0.0, 9.6296301], [10.74074, 20.74074], [2.2222223, 54.074074], [4.0740738, 50.740742], [34.444443, 46.296295], [11.481482, 1.4814816], [24.074076, -2.9629631], [74.814819, 79.259254], [67.777779, 152.22223], [57.037041, 127.03704], [89.259262, 12.222222]])

#     points = array([
#         [-0.5, -0.5],
#         [0.5, -0.5],
#         [-0.5, 0.5],
#         [0.5, 0.5],
#         [0.0, 0.0]])


#     points = array([
#         [-0.5, -0.5], [-0.5, 0.0],
#         [0.5, -0.5], [0.5, 0.0],
#         [-0.5, 0.5], [0.0, -0.5],
#         [0.5, 0.5], [0.0, 0.5],
#         [0.0, 0.0]])

#     voro = Voronoi(points, buffer_size=0.25)

#     voro.plot(tri=True)

#     print("---")
#     print(voro.points)
#     print("---")
#     print(voro.inner)
#     print("---")
#     print(voro.distance)
#     print("---")
#     print(voro.neighbor)
#     print("---")
#     print(voro.interface)
# test()
