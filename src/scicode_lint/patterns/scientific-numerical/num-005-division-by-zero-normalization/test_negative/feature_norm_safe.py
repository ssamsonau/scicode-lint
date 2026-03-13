import numpy as np


def normalize_features(X, eps=1e-8):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + eps)


def scale_to_unit_norm(vectors, eps=1e-8):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def percentage_change(old, new, eps=1e-10):
    return (new - old) / (np.abs(old) + eps) * 100
