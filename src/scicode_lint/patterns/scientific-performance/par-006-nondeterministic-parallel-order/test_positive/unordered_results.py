from multiprocessing import Pool

import numpy as np


def process_sample(x):
    return x**2 + 3 * x + 1


def parallel_compute(data):
    results = []
    with Pool(processes=4) as pool:
        for result in pool.imap_unordered(process_sample, data):
            results.append(result)
    return np.array(results)


input_data = np.arange(1000)
output = parallel_compute(input_data)
