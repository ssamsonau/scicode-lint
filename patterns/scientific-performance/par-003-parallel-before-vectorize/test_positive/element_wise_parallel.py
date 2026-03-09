import multiprocessing as mp

import numpy as np


def parallel_square(arr, num_workers=4):
    def square(x):
        return x**2

    with mp.Pool(num_workers) as pool:
        result = pool.map(square, arr.tolist())
    return np.array(result)


def parallel_normalize(data, num_workers=4):
    def normalize_element(x):
        return (x - data.mean()) / data.std()

    with mp.Pool(num_workers) as pool:
        result = pool.map(normalize_element, data.tolist())
    return np.array(result)
