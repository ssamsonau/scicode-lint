import numpy as np


def update_matrix_block(matrix):
    block = matrix[2:8, 3:7]
    block += 10
    return block


def normalize_segment(vector):
    segment = vector[20:80]
    mean_val = segment.mean()
    segment -= mean_val
    return segment


data = np.random.rand(10, 10)
modified = update_matrix_block(data)

vec = np.arange(100, dtype=float)
normed = normalize_segment(vec)
