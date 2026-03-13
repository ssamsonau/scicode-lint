import numpy as np


def compute_distances(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    return distances


coords = np.random.randn(500, 3)
dist_matrix = compute_distances(coords)
