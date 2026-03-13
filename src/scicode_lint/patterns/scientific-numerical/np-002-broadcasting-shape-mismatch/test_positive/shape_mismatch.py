import numpy as np


def apply_weights(images, weights):
    weighted = images * weights.reshape(1, 64, 1, 1)
    return weighted


def normalize_batch(data):
    batch_mean = data.mean(axis=1)
    normalized = data - batch_mean
    return normalized


images = np.random.rand(100, 64, 64, 3)  # (100, 64, 64, 3)
weights = np.random.rand(64, 1)
output = apply_weights(images, weights)

batch = np.random.rand(32, 128)  # (32, 128)
norm_batch = normalize_batch(batch)
