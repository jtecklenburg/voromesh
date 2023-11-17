# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:15:48 2022

@author: jante
"""
import numpy as np
import os
import toughio
import random

def strf(zahl):
    if zahl > 0:
        return np.format_float_scientific(zahl, 4).rjust(10)
    else:
        return np.format_float_scientific(zahl, 3).rjust(10)


def caption(name):

    s = "----1"
    sep = "----*----"

    for i in range(2, 9, 1):
        s = s + sep + str(i)

    return name + s


def elementnames(nel):

    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def num2base(number, base=35):
        res = ""
        while number:
            ind = number % base
            res = res + alphabet[ind]
            number = number // base
        return res[::-1] or "0"

    def toughname(number):
        (a, b) = divmod(number, 100)
        return num2base(a).zfill(3) + format(b, '02d')

    return [toughname(a) for a in range(nel)]


def _write_elements(file, names, volume, material, area, center):

    file.write(caption("ELEME")+'\n')

    for i in range(len(volume)):
        file.write(names[i] +                       # EL. NE -> Name
                   ''.rjust(10) +                   # NSEQ + NADD
                   material[i].ljust(5) +           # MA1, MA2 -> Material
                   strf(volume[i]) +                # VOLX
                   strf(area[i]) +                  # AHTX
                   strf(0) +                        # PMX
                   strf(center[i, 0]) +             # X
                   strf(center[i, 1]) +             # Y
                   strf(center[i, 2]) + '\n')       # Z

    file.write('\n')


def _write_conne(file, names, ne, isot, d1, d2, areax, betax):
    file.write(caption("CONNE")+'\n')

    for i in range(len(ne)):
        ele1 = names[ne[i][0]]
        ele2 = names[ne[i][1]]
        file.write(ele1+ele2 +                      # Name1, Name2
                   ''.rjust(15) +                   # NSEQ, NADD, NADS leer
                   str(int(isot[i])).rjust(5) +     # ISOT
                   strf(d1[i]) +                    # D1
                   strf(d2[i]) +                    # D2
                   strf(areax[i]) +                 # AREAX
                   strf(betax[i]) + '\n')           # BETAX
                                                    # SIGX

    file.write('\n')


def write_incon(path, names, incons):
    """
    Writes TOUGH3 INCON file
    INCON----1----*----2----*----3----*----4----*----5----*----6----*----7----*----8
    A11 0
    5.3141792000000e+07 1.5000000000000e-01 0.0000000000000e+00 8.3000000000000e+01
    A11 1
    5.3167536000000e+07 1.5000000000000e-01 0.0000000000000e+00 8.3000000000000e+01

    Parameters
    ----------
    path : str
        path to TOUGH input files. Filename for initial conditions is INCON
    names : Array of str of size (ne). ne is the number of cells
        DESCRIPTION.
    incons : Array of float of size (ne, 4)
        DESCRIPTION.

    Returns
    -------
    None.

    """

    filename = os.path.join(path, "INCON")

    f = open(filename, "w")
    f.write(caption("INCON")+'\n')

    for (name, incon) in zip(names, incons):

        f.write(name+'\n')
        i = [np.format_float_scientific(value, 8).rjust(20) for value in incon]
        for value in i:
            f.write(value)
        f.write('\n')

    f.write('\n')
    f.close()


def _update_gener(material, volume, wells, path2infile,
                  names=False, verbose=False):

    if type(names) == bool:
        names = np.array(elementnames(len(volume)))

    volume = np.array(volume)

    parameters = toughio.read_input(path2infile)
    gen = list()

    if verbose:
        print("---------------")

    for key in wells:
        ind = np.where(np.array(material) == key)
        well = wells[key]
        wellnames = names[ind]
        wellvolume = volume[ind]

        sumwellvolume = np.sum(wellvolume)

        for (cellname, cellvolume) in zip(wellnames, wellvolume):
            for com in well:

                rate = well[com]*cellvolume/sumwellvolume

                if verbose:
                    print(cellname.ljust(35) +
                          com.ljust(5) +
                          strf(rate))

                gen.append(
                    {
                        "label": cellname,
                        "type": com,
                        "rates": rate,
                    })

    if verbose:
        print("---------------")

    parameters["generators"] = gen
    toughio.write_input(os.path.join(path2infile), parameters)


def activate_pmx(path2infile):

    parameters = toughio.read_input(path2infile)
    parameters["rocks"]["SEED"] = {}
    toughio.write_input(os.path.join(path2infile), parameters)


# def write_foft(path, center, names=False):
#     mesh = toughio.read_mesh(os.path.join(path, "mesh.pickle"))

#     #Elemente für Zeitreihe plotten
#     label1 = mesh.labels[mesh.near((10.0, 0.0, 0.0))]
#     label2 = mesh.labels[mesh.near((100.0, 0.0, 0.0))]
#     parameters['element_history'] = [label1, label2]


def _write_mesh(path, volume, material, area, center, ne,
                isot, d1, d2, areax, betax, incon=False):
    """
    Writes TOUGH3 MESH file

    Parameters
    ----------
    filename : str
        path to mesh file. Filename for a mesh in TOUGH is "MESH"
    volume : Array of float of size (nele) with nele as number of cells
        Volume of cells.
    area : Array of float of size (nele)
        Area of each cell for heat transfer with semi-infinite confining beds.
    center : Array of float of size (nele, 3)
        coordinates of cell centers
    ne : Array of int of size (nint, 2)
        Position of the interface defined by two cell numbers
    isot : Array of int of size (nint)
        Flow directions. Values are 1, 2 or 3.
    d1 : Array of float of size (nint)
        Distance of center of first cell in ne to interface
    d2 : Array of float of size (nint)
        Distance of center of second cell in ne to interface
    areax : Array of float of size (nint) with nint as number of interfaces
        Area of interfaces between cells ne.

    Returns
    -------
    None.

    """

    filename = os.path.join(path, "MESH")

    f = open(filename, "w")

    names = elementnames(len(volume))
    _write_elements(f, names, volume, material, area, center)

    _write_conne(f, names, ne, isot, d1, d2, areax, betax)

    if type(incon) is not bool:
        write_incon(path, names, incon)

    f.close()
