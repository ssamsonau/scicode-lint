import numpy as np


def compute_distances_chunked(points, chunk_size=1000):
    n = len(points)
    results = []

    for i in range(0, n, chunk_size):
        chunk_end = min(i + chunk_size, n)
        chunk = points[i:chunk_end]

        diff = chunk[:, np.newaxis, :] - points[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))

        min_distances = np.min(distances, axis=1)
        results.append(min_distances)

    return np.concatenate(results)


coordinates = np.random.randn(50000, 3)
min_dists = compute_distances_chunked(coordinates)
