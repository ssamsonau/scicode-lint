import numpy as np


def check_convergence(current, target):
    if current == target:
        return True
    return False


def compute_and_check():
    a = 0.1 + 0.2
    if a == 0.3:
        return True
    return False


result = compute_and_check()
