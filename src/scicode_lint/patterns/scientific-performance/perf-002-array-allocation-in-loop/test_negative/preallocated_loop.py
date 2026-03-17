import numpy as np


def compute_rolling_stats(arr, window):
    n = len(arr) - window + 1
    means = np.empty(n)
    stds = np.empty(n)
    for i in range(n):
        window_data = arr[i : i + window]
        means[i] = window_data.mean()
        stds[i] = window_data.std()
    return means, stds


def aggregate_batch_results(batches):
    results = []
    for batch in batches:
        mean_val = np.mean(batch)
        results.append(mean_val)
    return np.array(results)


def vectorized_batch_process(data, batch_indices):
    return np.array([np.sum(data[start:end]) for start, end in batch_indices])
