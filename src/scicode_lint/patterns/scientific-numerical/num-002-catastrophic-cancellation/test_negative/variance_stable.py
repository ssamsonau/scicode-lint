import numpy as np


def compute_variance(data):
    return np.var(data, ddof=1)


def compute_covariance(x, y):
    return np.cov(x, y)[0, 1]


def compute_std(data):
    return np.std(data)


def normalize_distribution(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


samples = np.random.rand(1000)
var = compute_variance(samples)
