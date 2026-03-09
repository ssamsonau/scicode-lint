import numpy as np


def compute_entropy(probs):
    return -np.sum(probs * np.log(probs))


def cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred))


def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))
