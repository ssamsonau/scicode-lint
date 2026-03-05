import numpy as np


def compute_variance_stable(data):
    variance = np.var(data)
    return variance


def quadratic_discriminant(a, b, c):
    sqrt_disc = np.sqrt(np.abs(b**2 - 4 * a * c))
    return sqrt_disc


def relative_difference(value1, value2):
    if value2 == 0:
        return float("inf")
    rel_diff = (value1 - value2) / value2
    return rel_diff


data = np.random.rand(1000)
var = compute_variance_stable(data)

x = 10.5
y = 3.2
result = x - y
