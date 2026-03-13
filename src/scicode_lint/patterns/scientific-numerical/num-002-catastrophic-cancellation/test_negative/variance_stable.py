import numpy as np


def compute_variance(data):
    return np.var(data, ddof=1)


def quadratic_formula(a, b, c):
    discriminant = b**2 - 4 * a * c
    sqrt_disc = np.sqrt(discriminant)
    if b >= 0:
        x1 = (-b - sqrt_disc) / (2 * a)
        x2 = c / (a * x1)
    else:
        x1 = (-b + sqrt_disc) / (2 * a)
        x2 = c / (a * x1)
    return x1, x2
