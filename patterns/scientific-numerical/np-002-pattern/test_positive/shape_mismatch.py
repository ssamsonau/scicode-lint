import numpy as np


def apply_weights(images, weights):
    weighted = images * weights
    return weighted


def normalize_batch(data, means):
    normalized = data - means
    return normalized


def scale_features(matrix, scales):
    result = matrix / scales
    return result


images = np.random.rand(100, 64, 64, 3)
weights = np.random.rand(64, 1)
output = apply_weights(images, weights)

batch = np.random.rand(32, 128)
means = np.random.rand(32, 1)
norm_batch = normalize_batch(batch, means)
