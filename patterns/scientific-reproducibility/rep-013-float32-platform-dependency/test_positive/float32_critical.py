import numpy as np


def compute_hash(data):
    arr = np.array(data, dtype=np.float32)
    checksum = arr.sum()
    return int(checksum * 1e6)


def validate_results(expected, actual):
    expected_arr = np.array(expected, dtype=np.float32)
    actual_arr = np.array(actual, dtype=np.float32)
    return np.array_equal(expected_arr, actual_arr)


def accumulate_gradients(gradients):
    total = np.zeros(gradients[0].shape, dtype=np.float32)
    for g in gradients:
        total += g.astype(np.float32)
    return total
