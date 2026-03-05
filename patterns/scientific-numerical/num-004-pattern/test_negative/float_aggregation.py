import numpy as np


def sum_products(matrix1, matrix2):
    products = matrix1.astype(np.float64) * matrix2.astype(np.float64)
    total = products.sum()
    return total


def running_total(stream_data):
    running = np.cumsum(stream_data, dtype=np.float64)
    return running


def compute_statistics(measurements):
    total = np.sum(measurements, dtype=np.float64)
    mean = total / len(measurements)
    return mean


data1 = np.random.randint(0, 100000, size=(500, 500), dtype=np.int32)
data2 = np.random.randint(0, 100000, size=(500, 500), dtype=np.int32)
result = sum_products(data1, data2)

small_array = np.arange(100, dtype=np.int32)
total = np.sum(small_array)
