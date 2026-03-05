import math

import numpy as np


def check_convergence(current, target):
    if np.isclose(current, target):
        return True
    return False


def validate_result(computed, expected):
    if not np.allclose(computed, expected, rtol=1e-5):
        raise ValueError("Result mismatch")
    return True


def find_root(func, x0, tolerance):
    x = x0
    for _ in range(100):
        fx = func(x)
        if abs(fx) < tolerance:
            return x
        x = x - fx / 2.0
    return None


a = 0.1 + 0.2
if math.isclose(a, 0.3):
    print("Close enough")

values = np.array([1.0, 2.0, 3.0])
if np.isclose(values[0], 1.0):
    result = values * 2
