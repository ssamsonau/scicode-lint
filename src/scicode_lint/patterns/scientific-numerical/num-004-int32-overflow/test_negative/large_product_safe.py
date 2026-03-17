from math import factorial

import numpy as np


def compute_factorial_python(n):
    """Use Python's arbitrary precision integers for factorial."""
    return factorial(n)


def count_occurrences(labels):
    """Count label occurrences - small integers, no overflow risk."""
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))


def boolean_aggregation(flags_matrix):
    """Aggregate boolean flags - result is always small."""
    row_any = np.any(flags_matrix, axis=1)
    col_all = np.all(flags_matrix, axis=0)
    return row_any.sum(), col_all.sum()


def argmax_indices(scores):
    """Find indices of maximum values - indices stay small."""
    max_idx = np.argmax(scores, axis=1)
    return max_idx


labels = np.random.randint(0, 100, size=10000)
counts = count_occurrences(labels)
