import numpy as np


def apply_transform(data, factor=2):
    result = data * factor
    result = result + 10
    return result


def normalize_inplace(arr):
    result = arr.copy()
    result -= result.mean()
    result /= result.std()
    return result


class DataProcessor:
    def process(self, data):
        return np.log1p(data)
