import numpy as np


def process_data(data):
    subset = data[10:50]
    subset[:] = 0
    return subset


def filter_outliers(measurements):
    valid_range = measurements[5:95]
    valid_range *= 0.8
    return valid_range


def apply_window(signal):
    window_section = signal[100:200]
    for i in range(len(window_section)):
        window_section[i] = window_section[i] * 0.5
    return window_section


arr = np.ones(100)
result = process_data(arr)
