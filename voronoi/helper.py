# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:50:04 2023

@author: jante
"""

import numpy as np
from shapely.geometry import MultiPoint, Point, LineString, MultiPolygon
from shapely.ops import polygonize
from scipy.spatial import Voronoi, Delaunay
import time


def convex_hull(points, buffer_size):
    return MultiPoint([Point(i)
                       for i in points]).convex_hull.buffer(buffer_size)


def voronoi_bound(points, boundary):

    # Ermittle große Box um das zu verlegende Gebiet, damit die Punkte
    # am Rand ebenfalls eine Voronoi-Zelle haben.
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])

    dx = 1000*(xmax - xmin)
    dy = 1000*(ymax - ymin)

    xmin = xmin - dx
    xmax = xmax + dx
    ymin = ymin - dy
    ymax = ymax + dx

    points2 = np.append(points, [[xmax, ymax], [xmin, ymax],
                                 [xmax, ymin], [xmin, ymin]], axis=0)

    print("Voronoi...")
    start = time.time()

    vor = Voronoi(points2)

    print(time.time() - start)
    print("Shapely...")
    start = time.time()

    lines = [
        LineString(vor.vertices[line])
        for line in vor.ridge_vertices if -1 not in line
    ]

    result = MultiPolygon(
            [poly.intersection(boundary) for poly in polygonize(lines)])

    print(time.time() - start)
    print("Reorder...")
    start = time.time()

    # reordering of points, because voro results are in the wrong order
    # order = list()
    # point_list = dict()

    # for i, p in zip(np.arange(np.shape(points)[0]), points):
    #     point_list[i] = Point(p)

    # for v in result.geoms:
    #     i = 0
    #     for k in point_list.keys():
    #         if v.distance(point_list[k]) < 1e-8:
    #             order.append(k)
    #             point_list.pop(k)
    #             break

    # points = points[order]

    # order = []
    # point_list = np.empty(len(points), dtype=object)

    # for i, p in enumerate(points):
    #     point_list[i] = Point(p)

    # for v in result.geoms:
    #     mask = np.ones(len(point_list), dtype=bool)
    #     for k, point in enumerate(point_list):
    #         if v.distance(point) < 1e-8:
    #             order.append(k)
    #             mask[k] = False
    #             break
    #     point_list = point_list[mask]

    # points = points[order]

    print(time.time() - start)
    print("Done!")

    return result, points


def neighbors(points):

    tri = Delaunay(points)

    neigh = set()

    for n in tri.simplices:
        v1 = [n[0], n[1]]
        v2 = [n[1], n[2]]
        v3 = [n[2], n[0]]
        v1.sort()
        v2.sort()
        v3.sort()
        neigh.add(tuple(v1))
        neigh.add(tuple(v2))
        neigh.add(tuple(v3))

    return list(neigh)


def volumes(voro):
    return [r.area for r in voro.geoms]


def distance(points, neigh):

    nd = len(neigh)

    dist = np.zeros(nd)
    nvec = list()

    for n in range(nd):
        i = neigh[n][0]
        j = neigh[n][1]

        nv = points[i]-points[j]
        d = np.sqrt(np.sum((nv)**2))

        dist[n] = d
        nvec.append(nv/d)

    return dist, nvec


def innerinterfaces(ne, voro):
    inner = np.zeros(len(voro.geoms))

    for i, j in ne:
        inner[i] = inner[i] + 1
        inner[j] = inner[j] + 1

    return inner


def inbox(p, p1, p2, eps=1e-8):
    # gibt True zurück, wenn sich Punkt p in der Box befindet, die von
    # p1 und p2 aufgespannt wird.
    # Mit eps kann die Box vergrößert werden.

    x = [p1[0], p2[0]]
    y = [p1[1], p2[1]]

    x.sort()
    y.sort()

    x[0] = x[0] - eps
    x[1] = x[1] + eps
    y[0] = y[0] - eps
    y[1] = y[1] + eps

    return p[0] > x[0] and p[0] < x[1] and p[1] > y[0] and p[1] < y[1]


def normal(nvec, p1, p2, d=0, eps=10e-10):
    # Gibt True zurück, wenn der Normalenvektor von p1 zu p1 gleich dem
    # Vektor nvec ist. nvec ist normiert!
    # d ist die Länge des Normalenvektors.
    # Wenn sie nicht bekannt ist, wird sie beechnet.

    if d == 0:
        d = np.sqrt(np.sum((p1-p2)**2))

    vec = (p1-p2)/d

    # Normalenvektor von vec für 2D
    vec2 = np.zeros(2)
    vec2[0] = -vec[1]
    vec2[1] = vec[0]

    return np.all(np.abs(vec2+nvec) < eps) or np.all(np.abs(vec2-nvec) < eps)


def duplicates(values):

    visited = set()
    dup = {x for x in values if x in visited or (visited.add(x) or False)}

    return dup


def matchinterface(voro, ne):

    cells = [list(v.exterior.coords) for v in voro.geoms]


    interfaces = list()
    ne2 = list()

    # Für jede Nachbarschaft
    for nachbarn in ne:

        # Untersuche die Interfaces einer Zelle
        edges = cells[nachbarn[0]][:-1] + cells[nachbarn[1]][:-1]

        intersect = duplicates(edges)

        if len(intersect) == 2:
            [p1, p2] = intersect
            p1 = np.array(p1)
            p2 = np.array(p2)

            interfaces.append(np.sqrt(np.sum((p1-p2)**2)))
            ne2.append(nachbarn)

    return interfaces, ne2
