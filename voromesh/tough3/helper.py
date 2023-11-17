# -*- coding: utf-8 -*-

import numpy as np
import pyvista as pv
from numba import jit


def printProgress(iteration, total, totalmin=5000):
    """
    Call in a loop to show progress on terminal

    Parameters
    ----------
    iteration : Int
       Current iteration
    total : Int
        Total iterations
    totalmin : Int
        Minimum Iterations for showing progress
    """
    if total > totalmin:
        if iteration % int(total/100) == 0:
            print('\r'+str(int(iteration / int(total/100)))+"%", end="\r")

        # Print New Line on Complete
        if iteration == total:
            print()


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

def calc_conne(mesh):

    coords = mesh.points
    centers = mesh.cell_centers().points
    inner_interfaces = dict()  # area between two connected cells
    outer_interfaces = dict()  # outer surface of cell
    distances = dict()  # distance of centers between two connected cells
    betax = dict()
    g = np.array([0, 0, 1])

    for cell_id in np.arange(mesh.number_of_cells, dtype=int):

        p1 = mesh.get_cell(cell_id).point_ids
        cell_interfaces = find_cell_interfaces(p1)
        neigh = neighbors(mesh, cell_id)

        for n in neigh:
            p2 = mesh.get_cell(n).point_ids

            # common points of both cells define the interface
            interface = list(set(p1) & set(p2))

            # the interface is 2d, when the number of points is 3 or greater
            npoints = np.size(interface)
            if npoints > 2:  # 3D-Kante

                index = [i for i, j in enumerate(cell_interfaces)
                         if sorted(j) == sorted(interface)]

                # remove inner from interface to find outer interfaces
                interf = cell_interfaces.pop(index[0])

                # calculate interface area only once!
                if cell_id < n:
                    conne = [cell_id, n]
                    conne.sort()
                    key = tuple(conne)

                    inner_interfaces[key] = calc_interface_area(coords[interf, :], npoints)

                    # calc distance between centers
                    center1 = centers[cell_id]
                    center2 = centers[n]
                    v1 = center1 - center2

                    dist = np.linalg.norm(v1)
                    distances[key] = dist

                   # calc cos of connection between centers and gravity
                   # norm(g) = 1!
                    betax[key] = np.dot(v1, g) / (np.linalg.norm(v1))

        # calc area of outer interfaces
        area = 0
        for interf in cell_interfaces:

            npoints = np.size(interf)

            area += calc_interface_area(coords[interf, :], npoints)

        outer_interfaces[cell_id] = area

        printProgress(cell_id, mesh.number_of_cells)

    return distances, inner_interfaces, outer_interfaces, betax


@jit(nopython=True)
def calc_interface_area(nodelist, npoints):
    # https://en.wikipedia.org/wiki/Shoelace_formula#Generalization
    a = np.zeros(3)

    for i in np.arange(npoints):
        a += np.cross(nodelist[i, :],
                      nodelist[(i+1) % npoints, :])

    return np.linalg.norm(a)/2


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
