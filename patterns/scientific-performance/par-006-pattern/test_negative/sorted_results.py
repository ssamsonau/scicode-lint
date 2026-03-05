from multiprocessing import Pool

import numpy as np


def process_indexed_item(item):
    idx, value = item
    return idx, value**2 + 3 * value + 1


def parallel_compute_ordered(data):
    indexed_data = list(enumerate(data))
    with Pool(processes=4) as pool:
        results = pool.map(process_indexed_item, indexed_data)

    results.sort(key=lambda x: x[0])
    ordered_values = [val for idx, val in results]
    return np.array(ordered_values)


input_data = np.arange(1000)
output = parallel_compute_ordered(input_data)
