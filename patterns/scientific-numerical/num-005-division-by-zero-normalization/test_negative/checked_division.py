import numpy as np


def standardize(matrix):
    row_means = matrix.mean(axis=1, keepdims=True)
    row_stds = matrix.std(axis=1, keepdims=True)
    row_stds = np.maximum(row_stds, 1e-10)
    standardized = (matrix - row_means) / row_stds
    return standardized


def min_max_normalize(data, eps=1e-10):
    data_min = data.min()
    data_max = data.max()
    range_val = data_max - data_min
    normalized = (data - data_min) / (range_val + eps)
    return normalized


def coefficient_of_variation(samples):
    mean = samples.mean()
    std = samples.std()
    epsilon = 1e-12
    cv = std / (mean + epsilon)
    return cv


data = np.array([[1, 1, 1], [2, 3, 4], [5, 5, 5]])
std_data = standardize(data)
