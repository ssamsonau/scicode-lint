import numpy as np


def check_index(idx, max_idx):
    if idx == max_idx:
        return True
    return False


def count_matches(data, target_int):
    matches = data == target_int
    return np.sum(matches)


def validate_shape(matrix):
    rows, cols = matrix.shape
    if rows == cols:
        return True
    return False


counter = 0
if counter == 0:
    counter = 1

flags = np.array([1, 0, 1, 0], dtype=int)
if flags[0] == 1:
    print("Flag set")
