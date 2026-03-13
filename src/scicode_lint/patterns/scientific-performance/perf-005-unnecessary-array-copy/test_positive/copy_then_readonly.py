import numpy as np


def analyze_data(arr):
    data = arr.copy()
    mean_val = data.mean()
    std_val = data.std()
    max_val = data.max()
    return mean_val, std_val, max_val


def compute_statistics(measurements):
    copied = measurements.copy()
    total = np.sum(copied)
    average = np.mean(copied)
    variance = np.var(copied)
    return {"total": total, "average": average, "variance": variance}


def iterate_values(arr):
    data = arr.copy()
    results = []
    for val in data:
        results.append(val * 2)
    return results
