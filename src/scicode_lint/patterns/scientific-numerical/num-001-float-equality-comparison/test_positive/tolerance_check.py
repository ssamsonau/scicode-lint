import numpy as np


def check_normalized(vector):
    norm = np.linalg.norm(vector)
    return norm == 1.0


def verify_probability(probs):
    total = probs.sum()
    if total == 1.0:
        return True
    return False


def is_orthogonal(matrix):
    product = matrix @ matrix.T
    identity = np.eye(len(matrix))
    return (product == identity).all()
