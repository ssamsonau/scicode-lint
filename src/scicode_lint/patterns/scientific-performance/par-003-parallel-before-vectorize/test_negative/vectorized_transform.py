import numpy as np


def transform_array(arr):
    result = np.where(arr > 0, arr * 2, arr / 2)
    return result


data = np.random.randn(100000)
transformed = transform_array(data)
