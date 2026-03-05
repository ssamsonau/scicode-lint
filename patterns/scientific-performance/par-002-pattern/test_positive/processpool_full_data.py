from concurrent.futures import ProcessPoolExecutor

import numpy as np


def compute_statistics(arr):
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    return mean_val, std_val


def analyze_in_parallel(data_matrix):
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(compute_statistics, data_matrix) for _ in range(4)]
        results = [f.result() for f in futures]
    return results


large_matrix = np.random.randn(50000, 2000)
stats = analyze_in_parallel(large_matrix)
