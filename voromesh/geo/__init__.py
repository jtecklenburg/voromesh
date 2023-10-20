# -*- coding: utf-8 -*-
"""
Created on Wed May 31 17:14:39 2023

@author: jante
"""


import numpy as np
import pyvista as pv
import vtk


def update_z_from_surf(mesh, surf):

    z = surf.points[:, 2].copy()
    surf["z"] = z
    surf.points[:, 2] = 0

    mesh = mesh.sample(surf)
    mesh.points[:, 2] = mesh["z"]

    surf.points[:, 2] = z
    return mesh


def layersfromsurf(surfmesh, thickness):
    """
    Use a topographic surface to create a 3D terrain-following mesh.

    Parameters
    ----------
    surfmesh : Pyvista.UnstructuredGrid with 2D-Elements
        Surface mesh.
    thickness : List of float
        thickness of layers in z-direction.

    Returns
    -------
    mesh : Pyvista.UnstructuredGrid with 3D-Elements
        Extruded surface mesh.

    """

    surfmesh = surfmesh.compute_cell_sizes(length=False,
                                           area=True,
                                           volume=False)
    area = surfmesh["Area"]
    volume = list()

    # Typ des Geometry bzws. des Prismas
    ctype = {3: vtk.VTK_WEDGE,
             4: vtk.VTK_HEXAHEDRON,
             5: vtk.VTK_PENTAGONAL_PRISM,
             6: vtk.VTK_HEXAGONAL_PRISM,
             7: vtk.VTK_CONVEX_POINT_SET}

    nlayer = np.size(thickness)
    points = np.tile(surfmesh.points, (nlayer+1, 1))
    nc = surfmesh.number_of_points

    # Für jede Schicht werden z-Koordinaten festgelegt.
    for i in range(nlayer):
        points[((i+1)*nc):(i+2)*nc, 2] = (points[i*nc:((i+1)*nc), 2]
                                          + thickness[i])

    cells = list()
    celltypes = list()
    layer = list()

    ind = 0
    for i in range(surfmesh.number_of_cells):
        npoints = surfmesh.cells[ind]
        edges = surfmesh.cells[ind+1:ind+1+npoints]

        ind = ind + 1 + npoints

        for j in range(nlayer):
            celltypes.append(ctype.setdefault(npoints,
                                              vtk.VTK_CONVEX_POINT_SET))

            cells.append(2*npoints)         # number of points
            cells.extend(edges+j*nc)        # lower side
            cells.extend(edges+(j+1)*nc)    # upper side

            volume.append(area[i]*thickness[j])
            layer.append(j)

    mesh = pv.UnstructuredGrid(cells,
                               np.array(celltypes),
                               points)

    mesh["Volume"] = volume
    mesh["Layer"] = layer

    return mesh


def neighbors(mesh, cell_idx):
    """
    Find all neighbors of cell cell_idx with common nodes

    Parameters
    ----------
    mesh : pyvista.core.pointset.UnstructuredGrid
        mesh
    cell_idx : int
        number of cell in mesh

    Returns
    -------
    np.array
        list with idxs of cell neighbors.

    """
    cell = mesh.GetCell(cell_idx)
    pids = pv.vtk_id_list_to_array(cell.GetPointIds())
    neigh = set(mesh.extract_points(pids)["vtkOriginalCellIds"])
    neigh.discard(cell_idx)
    return np.array(list(neigh))


def find_cell_interfaces(points):
    """
    maps points to cell interfaces

    Parameters
    ----------
    points : list of int
        points of cell. Tested for prism-type celltypes only!

    Returns
    -------
    interf : list of list of int
        nodes for each interface.

    """

    points = np.array(points)
    n = int(np.size(points)/2)

    # the order of points is important for calculating the area.
    #  3------2
    #  |      |
    #  |      |   -> 0, 1, 2, 3
    #  0------1
    # clockwise or counterclockwise direction

    interf = list()
    interf.append(points[:n])
    interf.append(points[n:])

    for i in np.arange(n-1):
        idx = [i, (i+1) % n, (i+n+1) % (2*n), (i+n) % (2*n)]
        interf.append(points[idx])

    idx = [0, n-1, 2*n-1, n]
    interf.append(points[idx])

    return interf
