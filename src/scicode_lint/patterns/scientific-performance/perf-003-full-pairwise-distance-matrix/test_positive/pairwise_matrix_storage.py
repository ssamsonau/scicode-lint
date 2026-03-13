import numpy as np


def calculate_similarity_matrix(embeddings):
    num_samples = embeddings.shape[0]
    similarity = np.zeros((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(num_samples):
            similarity[i, j] = np.dot(embeddings[i], embeddings[j])

    return similarity


vectors = np.random.randn(30000, 512)
sim_matrix = calculate_similarity_matrix(vectors)
