import numpy as np


def vectorized_square(arr):
    return np.square(arr)


def vectorized_normalize(data):
    return (data - np.mean(data)) / np.std(data)
