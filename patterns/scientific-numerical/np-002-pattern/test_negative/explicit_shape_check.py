import numpy as np


def apply_weights(images, weights):
    assert images.shape[-1] == weights.shape[0]
    weighted = images * weights
    return weighted


def normalize_batch(data, means):
    if data.shape != means.shape:
        raise ValueError(f"Shape mismatch: {data.shape} vs {means.shape}")
    normalized = data - means
    return normalized


def scale_features(matrix, scales):
    assert matrix.shape[1] == len(scales)
    result = matrix / scales.reshape(1, -1)
    return result


images = np.random.rand(100, 64, 64, 3)
weights = np.random.rand(3)
output = apply_weights(images, weights)
