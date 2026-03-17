import numpy as np


def process_data(input_arr):
    working_copy = input_arr.copy()
    working_copy[:] = np.sort(working_copy)
    working_copy[0] = 0
    return working_copy


def accumulate(arr, accumulator):
    local = arr.copy()
    local += accumulator
    accumulator[:] = local
    return local
