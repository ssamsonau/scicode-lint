import numpy as np


def calculate_vector_norms(matrix):
    squared_matrix = matrix**2
    row_sums = np.sum(squared_matrix, axis=1)
    norms = np.sqrt(row_sums)
    return norms


large_matrix = np.random.randn(50000, 1000)
vector_norms = calculate_vector_norms(large_matrix)
