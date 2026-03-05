import numpy as np


def exponential_difference(x):
    val1 = np.exp(x)
    val2 = np.exp(x - 1e-8)
    diff = val1 - val2
    return diff


def compute_delta(measurements):
    baseline = measurements[0]
    delta = measurements[1] - baseline
    return delta


def distance_from_origin(point1, point2):
    dist1 = np.sqrt(point1[0] ** 2 + point1[1] ** 2 + point1[2] ** 2)
    dist2 = np.sqrt(point2[0] ** 2 + point2[1] ** 2 + point2[2] ** 2)
    diff = dist1 - dist2
    return diff


large_val = 1e15
small_delta = 1.0
result = (large_val + small_delta) - large_val
