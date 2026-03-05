import numpy as np


def compute_total(data):
    total = np.sum(data, dtype=np.int64)
    return total


def cumulative_sum(values):
    cumsum = np.cumsum(values, dtype=np.int64)
    return cumsum


def aggregate_counts(counts):
    total_count = counts.astype(np.int64).sum()
    return total_count


large_array = np.ones(1000000, dtype=np.int32) * 5000
total = compute_total(large_array)

counters = np.full(100000, 50000, dtype=np.int64)
result = aggregate_counts(counters)
