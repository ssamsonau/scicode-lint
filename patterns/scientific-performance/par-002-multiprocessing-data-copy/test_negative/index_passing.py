from multiprocessing import Pool

import numpy as np

big_array = np.random.randn(80000, 500)


def process_indices(idx_range):
    start, end = idx_range
    return np.sum(big_array[start:end])


def parallel_with_indices(num_workers):
    chunk_size = len(big_array) // num_workers
    index_ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_indices, index_ranges)

    return sum(results)


total_sum = parallel_with_indices(4)
