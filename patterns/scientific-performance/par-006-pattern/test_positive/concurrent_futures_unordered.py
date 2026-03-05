from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def transform_value(item):
    idx, val = item
    return np.sqrt(abs(val))


def batch_process(values):
    results = []
    indexed_values = list(enumerate(values))

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(transform_value, item): i for i, item in enumerate(indexed_values)
        }
        for future in as_completed(futures):
            results.append(future.result())

    return np.array(results)


data = np.random.randn(5000)
transformed = batch_process(data)
