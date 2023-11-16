import numpy as np
import pyvista as pv
import vtk

def update_z_from_surf(mesh, surf):
    """
    Maps elevation data from surf to mesh.

    Parameters
    ----------
    mesh : pyvista.UnstructuredGrid
        (Surface) Mesh without elevation data.
    surf : pyvista.UnstructuredGrid
        Surface mesh with elevation data.

    Returns
    -------
    mesh : pyvista.UnstructuredGrid
        (Surface) Mesh with elevation data from surf

    """

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
