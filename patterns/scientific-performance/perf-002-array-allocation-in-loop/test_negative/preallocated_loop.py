import numpy as np


def process_batches(data, batch_size):
    n_batches = len(data) // batch_size
    results = np.empty((n_batches, batch_size))
    for i in range(n_batches):
        batch = data[i * batch_size : (i + 1) * batch_size]
        results[i] = batch * 2
    return results


def compute_rolling_stats(arr, window):
    n = len(arr) - window + 1
    means = np.empty(n)
    stds = np.empty(n)
    for i in range(n):
        window_data = arr[i : i + window]
        means[i] = window_data.mean()
        stds[i] = window_data.std()
    return means, stds
