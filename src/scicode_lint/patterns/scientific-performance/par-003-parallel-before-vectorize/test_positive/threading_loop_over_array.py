from concurrent.futures import ThreadPoolExecutor

import numpy as np


def transform_element(value):
    return np.sin(value) + np.cos(value)


def parallel_element_transform(array, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(transform_element, array.tolist()))
    return np.array(results)


data = np.random.randn(10000)
output = parallel_element_transform(data)
