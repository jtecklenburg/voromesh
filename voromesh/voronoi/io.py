# -*- coding: utf-8 -*-

import vtk
from pyvista import UnstructuredGrid
from shapely.geometry import mapping
import numpy as np


def create_unstructured_mesh(points, connectivities):
    """
    Creates a VTK unstructured mesh from the given points and connectivities.
    Only polygon cells are supported.
    """
    # Create an empty VTK unstructured grid
    mesh = vtk.vtkUnstructuredGrid()

    # Create a VTK points object from the input points
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point)
    mesh.SetPoints(vtk_points)

    # Create a VTK cell array from the input connectivities
    vtk_cells = vtk.vtkCellArray()
    for connectivity in connectivities:
        n_points = len(connectivity)
        # print(n_points)
        vtk_cell = vtk.vtkPolygon()
        vtk_cell.GetPointIds().SetNumberOfIds(n_points)
        for j, point_index in enumerate(connectivity):
            vtk_cell.GetPointIds().SetId(j, point_index)
        vtk_cells.InsertNextCell(vtk_cell)

    mesh.SetCells(vtk.VTK_POLYGON, vtk_cells)

    return mesh


def add_dimension(points, dim="z"):

    z = np.zeros((np.shape(points)[0], 1))

    if dim.upper() == "Z" or dim.upper() == "XY":
        return np.hstack((points, z))
    elif dim.upper() == "X" or dim.upper() == "YZ":
        return np.hstack((z, points))
    elif dim.upper() == "Y" or dim.upper() == "XZ":
        return np.hstack((points[:, 0], z, points[:, 1]))
    else:
        raise ValueError("Parameter dim = " + str(dim) + " not implemented.")


def to_vtk(voronoi, decimalplace=8, dim="z"):

    # Find unique edges of geometries

    coord_list = list()
    n_cells = list()

    for geo in voronoi.geoms:
        mapped_geo = mapping(geo)
        coord = mapped_geo['coordinates'][0]
        coord_list = coord_list + list(coord)
        n_cells.append(np.shape(coord)[0])

    c_round = np.round(np.array(coord_list), decimalplace)

    coords_unique, ind = np.unique(c_round, return_inverse=True, axis=0)

    # Convert to VTK unstructured mesh
    points = add_dimension(coords_unique, dim)

    connectivities = list()
    i = 0

    for n in n_cells:

        # Polygone der unteren und der oberen Fläche des Prismas
        connectivities.append(list(ind[i:(i+n-1)]))
        i = i + n

    return create_unstructured_mesh(points, connectivities)


def to_pyvista(voronoi, decimalplace=8, dim="z"):
    vtkmesh = to_vtk(voronoi, decimalplace, dim)
    return UnstructuredGrid(vtkmesh)
