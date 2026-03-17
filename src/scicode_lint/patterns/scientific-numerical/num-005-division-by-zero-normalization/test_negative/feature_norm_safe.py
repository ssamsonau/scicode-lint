import numpy as np


def z_score_normalize(data, eps=1e-8):
    """Z-score normalization with epsilon guard."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / (std + eps)


def normalize_vectors(vectors, eps=1e-8):
    """L2 normalize vectors with epsilon guard to avoid division by zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + eps)


def minmax_normalize(data, eps=1e-8):
    """Min-max scaling with epsilon guard for constant features."""
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    range_val = max_val - min_val
    return (data - min_val) / (range_val + eps)


def feature_normalize_safe(X, eps=1e-8):
    """Normalize features using np.maximum to clamp denominator."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    safe_std = np.maximum(std, eps)
    return (X - mean) / safe_std


def normalize_embedding(x, eps=1e-8):
    """Normalize a single embedding vector with epsilon guard."""
    return x / (np.linalg.norm(x) + eps)


data = np.random.randn(100, 10)
normalized = z_score_normalize(data)
