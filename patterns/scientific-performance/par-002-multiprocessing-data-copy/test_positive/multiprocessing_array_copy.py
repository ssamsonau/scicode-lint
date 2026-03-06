from multiprocessing import Pool

import numpy as np


def apply_filter(data):
    filtered = data * 2.0
    result = np.mean(filtered)
    return result


def parallel_process_array(large_array, num_processes):
    with Pool(processes=num_processes) as pool:
        results = pool.map(apply_filter, [large_array] * num_processes)
    return np.array(results)


big_data = np.random.randn(100000, 1000)
outputs = parallel_process_array(big_data, 8)
