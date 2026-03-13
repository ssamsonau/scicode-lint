import numpy as np
from scipy.spatial.distance import cdist


def find_nearest(query_points, database_points, k=5):
    distances = cdist(query_points, database_points)
    nearest_indices = np.argsort(distances, axis=1)[:, :k]
    return nearest_indices


def compute_similarity_matrix(embeddings):
    n = len(embeddings)
    similarity = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity[i, j] = np.dot(embeddings[i], embeddings[j])
    return similarity
