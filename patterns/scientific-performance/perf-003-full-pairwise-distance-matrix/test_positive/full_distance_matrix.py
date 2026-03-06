import numpy as np
from scipy.spatial.distance import cdist


def compute_all_distances(points):
    _n = len(points)
    distance_matrix = cdist(points, points, metric="euclidean")
    return distance_matrix


coordinates = np.random.randn(50000, 3)
all_distances = compute_all_distances(coordinates)
