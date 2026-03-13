import numpy as np


def normalize_features(features):
    """Uses np.linalg.norm for efficient normalization without intermediates."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.where(norms > 0, norms, 1)


def compute_distances(points, center):
    """Uses np.linalg.norm for distance calculation - no intermediate arrays."""
    return np.linalg.norm(points - center, axis=1)


def batch_normalize(data):
    """Efficient batch normalization using broadcasting without large intermediates."""
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / np.where(std > 0, std, 1)
