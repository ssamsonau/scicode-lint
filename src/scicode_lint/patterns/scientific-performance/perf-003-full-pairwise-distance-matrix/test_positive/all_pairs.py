import numpy as np
from scipy.spatial.distance import cdist


def find_nearest_neighbors(query_embeddings, database_embeddings, k=5):
    full_distances = cdist(query_embeddings, database_embeddings)
    nearest_indices = np.argsort(full_distances, axis=1)[:, :k]
    nearest_distances = np.take_along_axis(full_distances, nearest_indices, axis=1)
    return nearest_indices, nearest_distances


queries = np.random.randn(10000, 256)
database = np.random.randn(50000, 256)
indices, distances = find_nearest_neighbors(queries, database, k=5)
