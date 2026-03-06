import numpy as np


def compute_normalized_distance(vectors):
    diff = vectors - np.mean(vectors, axis=0)
    squared = diff**2
    summed = np.sum(squared, axis=1)
    distances = np.sqrt(summed)
    return distances


data = np.random.randn(100000, 500)
result = compute_normalized_distance(data)
