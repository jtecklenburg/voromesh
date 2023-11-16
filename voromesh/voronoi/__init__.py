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
        Converts the Voronoi diagram to a vtk unstructured grid.

        Parameters
        ----------
        decimalplace : int, optional
            _description_. Defaults to 8.
        dim : str, optional
            _description_. Defaults to "Z".

        Returns:
            vtkUnstructuredGrid: VTK unstructured grid
        """

        return to_vtk(self.voronoi, decimalplace, dim)

    def to_pyvista(self, decimalplace=8, dim="Z"):
        """
        Converts the Voronoi diagram to a pyvista unstructured grid.

        Returns
        -------
            pyvista.UnstructuredGrid: Pyvista unstructured grid

        """
        return to_pyvista(self.voronoi, decimalplace, dim)
