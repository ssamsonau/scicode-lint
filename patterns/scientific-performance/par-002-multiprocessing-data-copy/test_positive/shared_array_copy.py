import multiprocessing as mp

import numpy as np


def process_chunks(data, num_workers=4):
    def worker(chunk):
        return chunk.sum()

    chunks = np.array_split(data, num_workers)
    with mp.Pool(num_workers) as pool:
        results = pool.map(worker, chunks)
    return sum(results)


def parallel_transform(large_array, transform_fn):
    with mp.Pool() as pool:
        results = pool.map(transform_fn, [large_array] * 4)
    return np.mean(results, axis=0)
