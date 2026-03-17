from functools import partial
from multiprocessing import Pool

import numpy as np


def compute_slice_stats(start_idx, end_idx, full_data):
    slice_data = full_data[start_idx:end_idx]
    return np.mean(slice_data), np.std(slice_data)


def parallel_slice_analysis(data, num_slices=8):
    slice_size = len(data) // num_slices
    ranges = [(i * slice_size, (i + 1) * slice_size) for i in range(num_slices)]

    worker_func = partial(compute_slice_stats, full_data=data)

    with Pool(processes=4) as pool:
        results = pool.starmap(worker_func, ranges)
    return results


large_dataset = np.random.randn(100000, 500)
stats = parallel_slice_analysis(large_dataset)
