import numpy as np


def normalize_features(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


def scale_to_unit_norm(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def percentage_change(old, new):
    return (new - old) / old * 100
