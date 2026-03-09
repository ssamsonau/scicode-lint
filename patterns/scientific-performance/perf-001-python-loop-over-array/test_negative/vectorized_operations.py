import numpy as np


def compute_distances(points, center):
    diff = points - center
    return np.sqrt(np.sum(diff**2, axis=1))


def normalize_rows(matrix):
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / row_sums
