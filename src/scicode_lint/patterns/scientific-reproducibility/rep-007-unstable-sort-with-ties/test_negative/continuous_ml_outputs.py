import numpy as np


def get_top_k_predictions(logits, k=5):
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    top_k_indices = np.argsort(-probs)[:k]
    return top_k_indices


def rank_by_similarity(query_embedding, document_embeddings):
    similarities = np.dot(document_embeddings, query_embedding)
    return np.argsort(-similarities)


def sort_by_confidence(model_outputs):
    confidence_scores = model_outputs.max(axis=1)
    return np.argsort(-confidence_scores)


def select_diverse_samples(embeddings, n_samples):
    distances = np.linalg.norm(embeddings[:, None] - embeddings, axis=2)
    diversity_scores = distances.min(axis=1)
    return np.argsort(-diversity_scores)[:n_samples]


def rank_by_log_probability(log_probs):
    return np.argsort(-log_probs)


def get_nearest_neighbors(query, database, k=10):
    distances = np.sqrt(np.sum((database - query) ** 2, axis=1))
    return np.argsort(distances)[:k]
