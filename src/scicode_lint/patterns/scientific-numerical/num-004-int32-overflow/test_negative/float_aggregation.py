import numpy as np


def normalized_dot_product(vec1, vec2):
    """Dot product of unit vectors - result bounded by 1."""
    v1_norm = vec1 / np.linalg.norm(vec1)
    v2_norm = vec2 / np.linalg.norm(vec2)
    return np.dot(v1_norm, v2_norm)


def percentage_calculation(counts, total):
    """Percentages are always 0-100, no overflow risk."""
    return 100.0 * counts / total


def rolling_average(data, window=10):
    """Rolling average maintains same scale as input."""
    cumsum = np.cumsum(np.insert(data.astype(float), 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def softmax(logits):
    """Softmax output is always in [0, 1], no overflow in result."""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


vec = np.random.rand(100)
sm = softmax(vec)
