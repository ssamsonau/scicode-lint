import numpy as np


def compute_total(data):
    total = np.sum(data)
    return total


def cumulative_sum(values):
    cumsum = np.cumsum(values)
    return cumsum


def aggregate_counts(counts):
    total_count = counts.sum()
    return total_count


large_array = np.ones(1000000, dtype=np.int32) * 5000
total = compute_total(large_array)

counters = np.full(100000, 50000, dtype=np.int32)
result = aggregate_counts(counters)
