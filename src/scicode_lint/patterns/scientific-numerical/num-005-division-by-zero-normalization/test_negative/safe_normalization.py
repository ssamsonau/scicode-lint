import numpy as np


def normalize_features(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized = (data - mean) / (std + 1e-8)
    return normalized


def z_score(values, eps=1e-8):
    mu = values.mean()
    sigma = values.std()
    scores = (values - mu) / (sigma + eps)
    return scores


def scale_to_unit_range(arr, eps=1e-8):
    min_val = arr.min()
    max_val = arr.max()
    range_val = max_val - min_val
    scaled = (arr - min_val) / (range_val + eps)
    return scaled


features = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
normed = normalize_features(features)
