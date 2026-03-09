import numpy as np


def check_normalized(vector, rtol=1e-5):
    norm = np.linalg.norm(vector)
    return np.isclose(norm, 1.0, rtol=rtol)


def verify_probability(probs, atol=1e-8):
    total = probs.sum()
    return np.isclose(total, 1.0, atol=atol)


def is_orthogonal(matrix, atol=1e-6):
    product = matrix @ matrix.T
    identity = np.eye(len(matrix))
    return np.allclose(product, identity, atol=atol)
