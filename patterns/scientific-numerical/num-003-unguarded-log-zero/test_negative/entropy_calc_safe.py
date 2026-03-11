import numpy as np


def compute_entropy(probs, eps=1e-10):
    safe_probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(safe_probs))


def cross_entropy(y_true, y_pred, eps=1e-10):
    safe_pred = np.clip(y_pred, eps, 1.0)
    return -np.mean(y_true * np.log(safe_pred))


def kl_divergence(p, q, eps=1e-10):
    safe_p = np.clip(p, eps, 1.0)
    safe_q = np.clip(q, eps, 1.0)
    return np.sum(safe_p * np.log(safe_p / safe_q))
