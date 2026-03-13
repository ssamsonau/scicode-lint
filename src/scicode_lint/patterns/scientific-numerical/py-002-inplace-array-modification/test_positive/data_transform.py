import numpy as np


def apply_transform(data, factor=2):
    data *= factor
    data += 10
    return data


def normalize_inplace(arr):
    arr -= arr.mean()
    arr /= arr.std()
    return arr


class DataProcessor:
    def process(self, data):
        data[:] = np.log1p(data)
        return data
