import numpy as np


def standardize(matrix):
    row_means = matrix.mean(axis=1, keepdims=True)
    row_stds = matrix.std(axis=1, keepdims=True)
    standardized = (matrix - row_means) / row_stds
    return standardized


def min_max_normalize(data):
    data_min = data.min()
    data_max = data.max()
    range_val = data_max - data_min
    normalized = (data - data_min) / range_val
    return normalized


def coefficient_of_variation(samples):
    mean = samples.mean()
    std = samples.std()
    cv = std / mean
    return cv


data = np.array([[1, 1, 1], [2, 3, 4], [5, 5, 5]])
std_data = standardize(data)

single_value = np.array([42.0, 42.0, 42.0])
normed = min_max_normalize(single_value)
