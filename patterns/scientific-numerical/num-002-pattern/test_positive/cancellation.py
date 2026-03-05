import numpy as np


def compute_variance_naive(data):
    mean_val = np.mean(data)
    mean_sq = np.mean(data**2)
    variance = mean_sq - mean_val**2
    return variance


def quadratic_discriminant(a, b, c):
    discriminant = b**2 - 4 * a * c
    return discriminant


def relative_difference(value1, value2):
    diff = value1 - value2
    rel_diff = diff / value2
    return rel_diff


data = np.random.rand(1000) * 1e10 + 1e12
var = compute_variance_naive(data)

x = 1.0000001
y = 1.0000000
result = x - y
