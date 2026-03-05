import numpy as np


def exponential_ratio(x):
    val1 = np.exp(x)
    val2 = np.exp(x - 1.0)
    ratio = val1 / val2
    return ratio


def compute_gradient(measurements):
    gradient = np.gradient(measurements)
    return gradient


def euclidean_distance(point1, point2):
    diff = point1 - point2
    dist = np.linalg.norm(diff)
    return dist


values = np.array([100.0, 50.0, 25.0])
differences = np.diff(values)
