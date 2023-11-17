# -*- coding: utf-8 -*-

import numpy as np
import pyvista as pv


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='',
                     decimals=1, length=80, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
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

        # p1 = mesh.cell_point_ids(cell_id)
        p1 = mesh.get_cell(cell_id).point_ids
        cell_interfaces = find_cell_interfaces(p1)
        neigh = neighbors(mesh, cell_id)

        for n in neigh:
            # p2 = mesh.cell_point_ids(n)
            p2 = mesh.get_cell(n).point_ids

            # common points of both cells define the interface
            seen = set()
            interface = [x for x in (p1+p2) if x in seen or seen.add(x)]

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

                    # calc interface area
                    a = np.zeros(3)
                    for i in np.arange(npoints):
                        # https://en.wikipedia.org/wiki/Shoelace_formula#Generalization

                        a = a + np.cross(coords[interf[i], :],
                                         coords[interf[(i+1) % npoints], :])

                    inner_interfaces[key] = np.linalg.norm(a)/2

                    # calc distance between centers
                    center1 = centers[cell_id]
                    center2 = centers[n]
                    dist = np.linalg.norm(center1 - center2)
                    distances[key] = dist

                    # calc cos of connection between centers and gravity
                    v1 = center1 - center2

                    # norm(g) = 1!
                    betax[key] = np.dot(v1, g) / (np.linalg.norm(v1))

        # calc area of outer interfaces
        area = 0
        for interf in cell_interfaces:

            a = np.zeros(3)
            npoints = np.size(interf)

            for i in np.arange(npoints):

                a = a + np.cross(coords[interf[i], :],
                                 coords[interf[(i+1) % npoints], :])

            area += np.linalg.norm(a)/2

        outer_interfaces[cell_id] = area

        printProgressBar(cell_id, mesh.number_of_cells)

    return distances, inner_interfaces, outer_interfaces, betax


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
