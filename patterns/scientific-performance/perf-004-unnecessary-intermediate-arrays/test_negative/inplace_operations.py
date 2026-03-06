import numpy as np


def normalize_inplace(matrix):
    mean_vals = np.mean(matrix, axis=0)
    matrix -= mean_vals
    std_vals = np.std(matrix, axis=0)
    matrix /= std_vals + 1e-8
    return matrix


data = np.random.randn(50000, 1000)
normalized = normalize_inplace(data.copy())
