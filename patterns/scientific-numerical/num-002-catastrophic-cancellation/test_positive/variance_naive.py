import numpy as np


def compute_variance(data):
    n = len(data)
    sum_sq = sum(x**2 for x in data)
    sum_x = sum(data)
    return (sum_sq - sum_x**2 / n) / (n - 1)


def quadratic_formula(a, b, c):
    discriminant = b**2 - 4 * a * c
    x1 = (-b + np.sqrt(discriminant)) / (2 * a)
    x2 = (-b - np.sqrt(discriminant)) / (2 * a)
    return x1, x2
