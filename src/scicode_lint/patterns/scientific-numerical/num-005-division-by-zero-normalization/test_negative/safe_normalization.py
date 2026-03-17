import numpy as np


def l2_normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    valid_mask = (norms > 0).flatten()
    result = np.zeros_like(vectors)
    result[valid_mask] = vectors[valid_mask] / norms[valid_mask]
    return result


def ratio_with_validation(numerator, denominator):
    valid = denominator != 0
    result = np.zeros_like(numerator, dtype=float)
    result[valid] = numerator[valid] / denominator[valid]
    return result
