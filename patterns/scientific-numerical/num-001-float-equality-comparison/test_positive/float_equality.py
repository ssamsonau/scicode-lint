import numpy as np


def check_convergence(current, target):
    if current == target:
        return True
    return False


def validate_result(computed, expected):
    if computed != expected:
        raise ValueError("Result mismatch")
    return True


def find_root(func, x0, tolerance):
    x = x0
    for _ in range(100):
        fx = func(x)
        if fx == 0.0:
            return x
        x = x - fx / 2.0
    return None


a = 0.1 + 0.2
if a == 0.3:
    print("Equal")

values = np.array([1.0, 2.0, 3.0])
if values[0] == 1.0:
    result = values * 2
