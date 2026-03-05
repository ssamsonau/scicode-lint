import numpy as np


def compute_distances(points, centers):
    diffs = points - centers
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    return distances


def apply_transformation(data, transform_matrix):
    result = data @ transform_matrix
    return result


pts = np.random.rand(1000, 3)
ctrs = np.random.rand(10, 3)
dists = compute_distances(pts, ctrs)

data = np.random.rand(50, 100)
trans = np.random.rand(50, 50)
transformed = apply_transformation(data, trans)
