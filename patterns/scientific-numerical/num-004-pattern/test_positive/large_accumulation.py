import numpy as np


def sum_products(matrix1, matrix2):
    products = matrix1 * matrix2
    total = products.sum()
    return total


def running_total(stream_data):
    running = np.cumsum(stream_data)
    return running


def compute_statistics(measurements):
    total = np.sum(measurements)
    mean = total / len(measurements)
    return mean


data1 = np.random.randint(0, 100000, size=(500, 500), dtype=np.int32)
data2 = np.random.randint(0, 100000, size=(500, 500), dtype=np.int32)
result = sum_products(data1, data2)

stream = np.arange(1000000, dtype=np.int32)
totals = running_total(stream)
