import numpy as np


def compute_log_sum(data):
    result = np.log1p(data)
    return result


def compute_log_ratio(numerator, denominator):
    eps = 1e-12
    ratio = numerator / (denominator + eps)
    log_ratio = np.log(np.maximum(ratio, eps))
    return log_ratio


def information_gain(prior, posterior):
    nonzero = posterior > 0
    log_ratio = np.zeros_like(posterior)
    log_ratio[nonzero] = np.log(posterior[nonzero] / prior[nonzero])
    gain = np.sum(posterior * log_ratio)
    return gain


values = np.array([0.0, 0.1, 0.5, 1.0])
result = compute_log_sum(values)
