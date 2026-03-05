import numpy as np


def compare_arrays(arr1, arr2):
    if (arr1 == arr2).all():
        return True
    return False


def detect_change(old_state, new_state):
    changed = old_state != new_state
    return changed


def verify_optimization(weights_before, weights_after):
    if np.sum(weights_before) == np.sum(weights_after):
        return False
    return True


x = np.array([0.1, 0.2, 0.3])
y = x + 1e-15
match = compare_arrays(x, y)

params = np.array([1.5, 2.7, 3.9])
if params[0] != 1.5:
    params[0] = 1.5
