import numpy as np


def get_stats(data):
    subset = data[10:50]
    mean_value = subset.mean()
    std_value = subset.std()
    return mean_value, std_value


def find_max_segment(measurements):
    segment = measurements[5:95]
    max_idx = segment.argmax()
    return max_idx


def compute_energy(signal):
    window = signal[100:200]
    energy = np.sum(window**2)
    return energy


arr = np.random.rand(100)
stats = get_stats(arr)
