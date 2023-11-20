import numpy as np
from .helper import calc_conne
from .tough3 import _write_mesh, _update_gener

import toughio, os
import pyvista as pv
from pvdwriter import pvdwriter
import sys


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


def outfile2vtu(path, mesh, outfile="OUTFILE", do_sort=True):
    """
    Reads an OUTFILE from a TOUGH3 simulation and append simulation results
    to the vtu mesh for each timestamp in the OUTFILE. Also writes a pvd file
    to relate files and timestamps in Paraview.

    Parameters
    ----------
    path : string
        path to TOUGH3 OUTFILE
    mesh : string
        name of vtu mesh in path
    outfile : string
        name of outfile. Default = "OUTFILE".
    do_sort : boolean
        parallel simulations may change the order of elements. For this
        reason sorting may be a good idea. Default = True.

    Returns
    -------
    None.

    """

    try:
        path_outfile = os.path.join(path, outfile)
        outputs = toughio.read_output(path_outfile)
    except OSError:
        print("Error reading OUTFILE: " + path_outfile)
        sys.exit()

    try:
        path_meshfile = os.path.join(path, mesh)
        mesh = pv.read(path_meshfile)
    except OSError:
        print("Error reading mesh file: " + path_meshfile)
        sys.exit

    pvdfile = os.path.join(path, "outfile.pvd")
    pvd = pvdwriter(pvdfile)

    # Parallel simulations may change the order of elements.
    if do_sort:
        nn = outputs[-1][3]
        ind = np.argsort(nn)

    i = 0

    for output in outputs:
        for key in output[4]:

            if do_sort:
                mesh[key] = output[4][key][ind]
            else:
                mesh[key] = output[4][key]


        timestamp = output[2]/(365.25*24*60*60)
        filename = "mesh_" + str(i) + ".vtu"
        print(filename + " | "+ str(timestamp))

        pvd.append(filename, timestamp)

        mesh.save(os.path.join(path, filename))
        i = i + 1

    pvd.close()
    print("complete!")
