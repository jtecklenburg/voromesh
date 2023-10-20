# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:50:04 2023

@author: jante
"""

import numpy as np
from shapely.geometry import MultiPoint, Point, LineString, MultiPolygon
from shapely.ops import polygonize
from scipy.spatial import Voronoi


def convex_hull(points, buffer_size):
    return MultiPoint([Point(i)
                       for i in points]).convex_hull.buffer(buffer_size)

    ### points = MultiPoint([(0.0, 0.0), (1.0, 1.0)])
    ### shapely.ops.triangulate(geom, tolerance=0.0, edges=False)
    ### shapely.ops.voronoi_diagram(geom, envelope=None, tolerance=0.0, edges=False)
    ### shapely.ops.unary_union(geoms)
    ### object.buffer(distance

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

    vor = Voronoi(points2)

    lines = [
        LineString(vor.vertices[line])
        for line in vor.ridge_vertices if -1 not in line
    ]

    result = MultiPolygon(
            [poly.intersection(boundary) for poly in polygonize(lines)])

    return result, points
