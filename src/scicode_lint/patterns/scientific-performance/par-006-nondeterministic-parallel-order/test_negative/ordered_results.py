import multiprocessing as mp

import numpy as np


def process_items(items, num_workers=4):
    def process(item):
        return item**2

    with mp.Pool(num_workers) as pool:
        results = pool.map(process, items)
    return np.sum(results)


def aggregate_ordered(data_chunks):
    with mp.Pool() as pool:
        results = pool.map(np.mean, data_chunks)
    return results
