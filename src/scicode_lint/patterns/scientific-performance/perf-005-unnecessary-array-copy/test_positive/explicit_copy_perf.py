import numpy as np


def normalize_data(arr):
    data = arr.copy()
    data = data - data.mean()
    data = data / data.std()
    return data


def process_array(input_arr):
    working = np.array(input_arr)
    working = working * 2
    working = working + 1
    return working
