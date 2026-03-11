import numpy as np


def normalize_rows(data):
    """Normalize each row by subtracting row mean."""
    row_means = data.mean(axis=1)
    return data - row_means


def standardize_samples(X):
    """Standardize each sample (row)."""
    sample_means = X.mean(axis=1)
    sample_stds = X.std(axis=1)
    return (X - sample_means) / sample_stds
