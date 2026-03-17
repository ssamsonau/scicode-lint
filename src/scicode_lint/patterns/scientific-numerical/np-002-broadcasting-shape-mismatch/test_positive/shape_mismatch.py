import numpy as np


def normalize_batch(data):
    batch_mean = data.mean(axis=1)
    normalized = data - batch_mean
    return normalized


batch = np.random.rand(32, 128)
norm_batch = normalize_batch(batch)
