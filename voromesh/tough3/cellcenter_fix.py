import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay

def calculate_tetrahedron_volume(points):
    return np.abs(np.linalg.det(points[1:] - points[0])) / 6

def calculate_cell_center(cell):
    points = cell.points
    tri = Delaunay(points)
    centroids = np.array([np.mean(points[simplex], axis=0) for simplex in tri.simplices])
    volumes = np.array([calculate_tetrahedron_volume(points[simplex]) for simplex in tri.simplices])
    return np.average(centroids, weights=volumes, axis=0)

def calculate_cell_centers(grid):
    centers = grid.cell_centers().points
    for cell_type, i in zip(grid.celltypes, range(grid.n_cells)):
        if cell_type == 41: # VTK doesn't calculate the centers for cell_type 41 correctly
            cell = grid.extract_cells(i)
            centers[i] = calculate_cell_center(cell)
    return centers

