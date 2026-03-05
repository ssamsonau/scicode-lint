import numpy as np


def normalize_array(arr):
    max_val = np.max(arr)
    result = np.where(arr > 0, arr / max_val, 0)
    return result


data = np.random.randn(100000)
normalized = normalize_array(data)
