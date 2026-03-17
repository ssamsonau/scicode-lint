import numpy as np
from scipy import stats


def pairwise_distances_stable(points):
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            distances[i, j] = d
            distances[j, i] = d
    return distances


def compute_z_scores(data):
    mean = np.mean(data)
    std = np.std(data)
    if std > 0:
        return (data - mean) / std
    return np.zeros_like(data)


def running_exponential_mean(data, alpha=0.1):
    result = np.empty_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result


points = np.random.rand(50, 3)
dist_matrix = pairwise_distances_stable(points)
