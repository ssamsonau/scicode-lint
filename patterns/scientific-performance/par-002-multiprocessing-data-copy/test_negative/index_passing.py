from concurrent.futures import ProcessPoolExecutor

import numpy as np

global_data = None


def init_worker(data):
    global global_data
    global_data = data


def process_indices(idx_range):
    start, end = idx_range
    chunk_result = np.sum(global_data[start:end])
    return chunk_result


def parallel_with_indices(data, num_workers):
    chunk_size = len(data) // num_workers
    index_ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]

    with ProcessPoolExecutor(
        max_workers=num_workers, initializer=init_worker, initargs=(data,)
    ) as executor:
        results = list(executor.map(process_indices, index_ranges))

    return sum(results)


big_array = np.random.randn(80000, 500)
total_sum = parallel_with_indices(big_array, 4)
