import numpy as np


def normalize_features(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized = (data - mean) / std
    return normalized


def z_score(values):
    mu = values.mean()
    sigma = values.std()
    scores = (values - mu) / sigma
    return scores


def scale_to_unit_range(arr):
    min_val = arr.min()
    max_val = arr.max()
    scaled = (arr - min_val) / (max_val - min_val)
    return scaled


features = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
normed = normalize_features(features)

constant_data = np.ones(100)
scores = z_score(constant_data)
