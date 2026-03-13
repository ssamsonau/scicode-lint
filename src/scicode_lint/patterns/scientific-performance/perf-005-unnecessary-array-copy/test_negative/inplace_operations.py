import numpy as np


def normalize_inplace(arr):
    data = arr.copy()
    data -= data.mean()
    data /= data.std()
    return data


def sort_copy(arr):
    sorted_arr = arr.copy()
    sorted_arr.sort()
    return sorted_arr


def fill_zeros(arr, threshold):
    result = arr.copy()
    result[result < threshold] = 0
    return result
