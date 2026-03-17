import numpy as np


def pca_projection(data, n_components=2):
    centered = data - data.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    return centered @ eigenvectors[:, idx]


def cosine_similarity_matrix(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    return normalized @ normalized.T


def elementwise_confidence_interval(samples):
    mean = samples.mean(axis=0)
    std = samples.std(axis=0, ddof=1)
    margin = 1.96 * std / np.sqrt(len(samples))
    return mean - margin, mean + margin


def softmax_rows(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)
