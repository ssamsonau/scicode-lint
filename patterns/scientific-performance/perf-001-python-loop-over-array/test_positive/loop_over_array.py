import numpy as np


def normalize_array(arr):
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        if arr[i] > 0:
            result[i] = arr[i] / np.max(arr)
        else:
            result[i] = 0
    return result


data = np.random.randn(100000)
normalized = normalize_array(data)
