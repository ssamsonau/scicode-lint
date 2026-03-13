import numpy as np


def compute_factorial(n):
    result = np.int64(1)
    for i in range(2, n + 1):
        result *= np.int64(i)
    return result


def sum_large_array(arr):
    return np.array(arr, dtype=np.int64).sum()


def matrix_product_sum(A, B):
    product = np.dot(A.astype(np.int64), B.astype(np.int64))
    return product.sum()
