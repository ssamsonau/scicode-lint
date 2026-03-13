import numpy as np


def check_convergence(loss, threshold=1e-7):
    return loss == threshold


def validate_output(result, expected):
    return np.float32(result) == np.float32(expected)


def compute_and_compare(a, b, target):
    result = np.float32(a) * np.float32(b)
    return result == target
