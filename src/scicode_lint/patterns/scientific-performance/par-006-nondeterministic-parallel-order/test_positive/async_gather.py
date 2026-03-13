import multiprocessing as mp

import numpy as np


def process_items(items, num_workers=4):
    def process(item):
        return item**2

    with mp.Pool(num_workers) as pool:
        results = list(pool.imap_unordered(process, items))
    return np.sum(results)


def aggregate_async(data_chunks):
    results = []

    def callback(result):
        results.append(result)

    with mp.Pool() as pool:
        for chunk in data_chunks:
            pool.apply_async(np.mean, (chunk,), callback=callback)
        pool.close()
        pool.join()
    return results
