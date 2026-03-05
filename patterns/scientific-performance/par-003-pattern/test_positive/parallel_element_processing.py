from multiprocessing import Pool

import numpy as np


def process_element(x):
    if x > 0:
        return x * 2
    else:
        return x / 2


def parallel_transform(arr):
    with Pool(processes=4) as pool:
        result = pool.map(process_element, arr.tolist())
    return np.array(result)


data = np.random.randn(100000)
transformed = parallel_transform(data)
