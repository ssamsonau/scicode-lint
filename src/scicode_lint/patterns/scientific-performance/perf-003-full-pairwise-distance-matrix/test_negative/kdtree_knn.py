import numpy as np
from scipy.spatial import cKDTree


def find_nearest_neighbor_distances(points, k=5):
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k + 1)
    return distances[:, 1]


coordinates = np.random.randn(50000, 3)
nearest_dists = find_nearest_neighbor_distances(coordinates)
