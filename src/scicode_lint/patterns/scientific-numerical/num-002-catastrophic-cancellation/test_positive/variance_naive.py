import numpy as np


def exp_nearly_equal_diff(a, b):
    """Subtracting exp of nearly-equal values causes catastrophic cancellation.

    When a ≈ b, exp(a) ≈ exp(b) and nearly all precision is lost in the subtraction.
    """
    result = np.exp(a) - np.exp(b)
    return result


def derivative_centered_diff(f, x, h=1e-10):
    """Centered difference with tiny h causes cancellation.

    f(x+h) and f(x-h) are nearly equal, their difference loses precision.
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def covariance_two_pass(x, y):
    """Two-pass covariance formula prone to cancellation with large means."""
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum(x[i] * y[i] for i in range(n)) / n - mean_x * mean_y
    return cov
