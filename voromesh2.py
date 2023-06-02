# -*- coding: utf-8 -*-
"""
Created on Wed May 31 17:14:39 2023

@author: jante
"""

import pickle
import numpy as np
from os.path import join
import pyvista as pv
import vtk

from voronoi import Voronoi


testcase = 0
plotvoro = False
plotmesh = False
fromfile = True

plotvoro = True
plotmesh = True
# fromfile = False


def update_z_from_surf(mesh, surf):

    z = surf.points[:, 2]
    surf["z"] = z
    surf.points[:, 2] = 0
    mesh = mesh.sample(surf)
    mesh.points[:, 2] = mesh["z"]
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

def calc_conne(mesh):

    coords = mesh.points
    centers = mesh.cell_centers().points
    inner_interfaces = dict()  # area between two connected cells
    outer_interfaces = dict()  # outer surface of cell
    distances = dict()  # distance of centers between two connected cells
    betax = dict()
    g = np.array([0, 0, 1])

    for cell_id in np.arange(mesh.number_of_cells, dtype=int):

        p1 = mesh.cell_point_ids(cell_id)
        cell_interfaces = find_cell_interfaces(p1)
        neigh = neighbors(mesh, cell_id)

        for n in neigh:
            p2 = mesh.cell_point_ids(n)

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

        #printProgressBar(cell_id, mesh.number_of_cells)

    return distances, inner_interfaces, outer_interfaces, betax



path = r"C:\Python\packages\voromesh\testdata"

if testcase == 0:    # Kreisförmiges Voronoi-Mesh
    points = pickle.load(open(join(path, "mesh_points.pickle"), "rb"))
    path = join(path, "voro_rand3")
    radius = 400
    thickness = np.array([4.5] * 10)
elif testcase == 1:   # Uniform mesh -> sgrid2
    points = pickle.load(open(join(path, "uniform_points.pickle"), "rb"))
    path = join(path, "sgrid_voro")
    radius = 50.5
    thickness = np.array([4.5] * 10)
else:
    points = pickle.load(open(join(path, "points.p"), "rb"))
    points = np.delete(points, 2, 1)
    path = join(path, "points")
    radius = 0.1
    thickness = [0.10, 0.20, 0.30]


voro = Voronoi(points, buffer_size=radius)

if plotvoro:
    voro.plot()

surfmesh = voro.to_pyvista()

# 2. Mapping

vtk1 = r"C:\Test\features\Local Server\TUNB\11_Basis_Mittlerer_Buntsandstein\11_ZNS_sm.vtk"
surf = pv.read(vtk1)

surfmesh = update_z_from_surf(surfmesh, surf)

surfmesh = surfmesh.compute_cell_sizes(length=False, area=True, volume=False)
#area = surfmesh["Area"]

if plotmesh:
    print(surfmesh.array_names)
    surfmesh.plot(show_edges=True, scalars="z")


area = surfmesh["Area"]

mesh = layersfromsurf(surfmesh, thickness)

print("Berechne Interfaces zwischen den Zellen...")
#dist, areai, areao, betax = calc_conne(mesh)

mesh["z"] = mesh.points[:, 2]

mesh.plot(show_edges=True)
