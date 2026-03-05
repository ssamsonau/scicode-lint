import numpy as np


def compute_normalized_distance(vectors):
    centered = vectors - np.mean(vectors, axis=0)
    distances = np.linalg.norm(centered, axis=1)
    return distances


data = np.random.randn(100000, 500)
result = compute_normalized_distance(data)
