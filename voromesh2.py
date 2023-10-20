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
from io_tough import write_mesh2, update_gener2

testcase = 1
plotvoro = False
plotmesh = False
fromfile = True

# plotvoro = True
# plotmesh = True
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

mesh["z"] = mesh.points[:, 2]

if plotmesh:
    mesh.plot(show_edges=True)


# %% Materialien

mesh.cell_data["material"] = 1

# well
xwell = 391677.41
ywell = 6073156.27
ind = mesh.find_cells_along_line([xwell, ywell, -3000], [xwell, ywell, 0])
mesh.cell_data["material"][ind] = 2

materials = {1: "SAND",
             2: "WELL"}

# %% Anfangsbedingungen
SSalt = 0.15
SCO2 = 0.0
p0 = 29430000
T0 = 83

centers = mesh.cell_centers().points
incon = np.full((mesh.n_cells, 4), -1.0e9)
incon[:, 0] = p0 - 9810.0 * centers[:, 2]
incon[:, 1] = SSalt
incon[:, 2] = SCO2
incon[:, 3] = T0

mesh.cell_data["initial_condition"] = incon

# mesh2.write_tough(os.path.join(path, "MESH"), incon=True)
# mesh2.write(os.path.join(path, "mesh.pickle"))

wells = {"WELL": {"COM3": 28.53,
                  "COM1": 1e-6}}

mesh.save(join(path, "mesh.vtu"))

write_mesh2(join(path), mesh, materials)
update_gener2(join(path, "INFILE"), mesh, materials, wells)
