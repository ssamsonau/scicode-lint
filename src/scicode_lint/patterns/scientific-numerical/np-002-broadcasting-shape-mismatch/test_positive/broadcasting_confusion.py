import numpy as np


def batch_normalize(batch):
    """Normalize each sample in batch."""
    batch_mean = batch.mean(axis=1)
    batch_std = batch.std(axis=1)
    normalized = (batch - batch_mean) / batch_std
    return normalized


def center_rows(matrix):
    """Center each row by subtracting row mean."""
    row_means = matrix.mean(axis=1)
    return matrix - row_means
