import numpy as np


def compute_distances(points, center):
    distances = []
    for point in points:
        dist = np.sqrt(sum((p - c) ** 2 for p, c in zip(point, center)))
        distances.append(dist)
    return distances


def normalize_rows(matrix):
    result = np.zeros_like(matrix)
    for i in range(len(matrix)):
        row_sum = sum(matrix[i])
        for j in range(len(matrix[i])):
            result[i, j] = matrix[i, j] / row_sum
    return result
