import numpy as np
from .helper import calc_conne
from .tough3 import _write_mesh, _update_gener

def write_mesh(path, mesh, materials):
    """
    Writes a mesh in TOUGH2/TOUGH3 format.

    Parameters
    ----------
    path : string
        Path for writing the TOUGH3 mesh.
    mesh : pyvista.UnstructuredGrid with cell data (see below)
        mesh["Volume"] :  cell volumes
        mesh["material"] : cell material as integer
        mesh["initial_condition"] : cell initial conditions as list
                                    find more information in TOUGH EOS manuals
    materials : dict
        Relate numbers in mesh["material"] to material names from the TOUGH
        INFILE. Example: materials = {1: "WELL", 2: "SAND"}

    Returns
    -------
    None.

    """
    dist, areai, areao, betax_ = calc_conne(mesh)

    volume = mesh["Volume"]                             # 1
    material = [materials[i] for i in mesh["material"]]  # 2

    area = np.zeros(np.size(volume))
    for key in areao:
        area[key] = areao[key]                          # 3

    centers = mesh.cell_centers().points                # 4

    ne = list(areai.keys())                             # 5

    d1 = 0.5 * np.fromiter(dist.values(), dtype=float)  # 7
    d2 = d1                                             # 8

    areax = np.fromiter(areai.values(), dtype=float)    # 9

    betax = np.fromiter(betax_.values(), dtype=float)   # 10

    incon = mesh["initial_condition"]                   # 11

    # Nur Unterschiede in h (1,2 -> 1) und v (3)
    isot = np.ones(np.size(areax))                      # 6
    ind = np.abs(betax) > 0.5
    isot[ind] = 3

    _write_mesh(path, volume, material, area, centers,
                ne, isot, d1, d2, areax, betax, incon)


def update_gener(path2infile, mesh, materials, wells):
    """
    Update the well information in the TOUGH3 INFILE

    Parameters
    ----------
    path2infile : string
        path to TOUGH3 INFILE.
    mesh : pyvista.UnstructuredGrid with cell data
        see write_mesh function.
    materials : dict
        see write_mesh function.
    wells : dict
        Define inflow and outflows.
        Key of dict is the material. The flow is splitted according to the
        volume fraction, when more than one cell has the same material.
        Value of dict is a dict with flow components. See TOUGH3 manual and
        TOUGH3 EOS manuals for details.
        Example: wells = {"WELL": {"COM1": 28.53, "COM3": 1e-6}}


    Returns
    -------
    None.

    """

    material = [materials[i] for i in mesh["material"]]
    volume = mesh["Volume"]

    _update_gener(material, volume, wells, path2infile,
                  names=False, verbose=False)
