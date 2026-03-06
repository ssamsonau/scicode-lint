import numpy as np


def information_gain(prior, posterior):
    log_ratio = np.log(posterior / prior)
    gain = np.sum(posterior * log_ratio)
    return gain


def apply_log_scale(measurements):
    log_scaled = np.log(measurements)
    return log_scaled


def kl_divergence(p, q):
    kl = np.sum(p * np.log(p / q))
    return kl


data = np.random.rand(100)
data[data < 0.01] = 0
log_data = apply_log_scale(data)

p_dist = np.array([0.1, 0.4, 0.5, 0.0])
q_dist = np.array([0.2, 0.3, 0.5, 0.0])
kl = kl_divergence(p_dist, q_dist)
