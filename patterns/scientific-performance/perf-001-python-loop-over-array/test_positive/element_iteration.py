import numpy as np


def compute_distances(points):
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt(np.sum((points[i] - points[j]) ** 2))
    return distances


coords = np.random.randn(500, 3)
dist_matrix = compute_distances(coords)
