import numpy as np


def accumulate_results(data_generator, processor):
    all_results = np.array([])
    for data in data_generator:
        new_result = processor(data)
        all_results = np.concatenate([all_results, new_result])
    return all_results


def build_matrix_rowwise(rows):
    matrix = np.empty((0, 100))
    for row_data in rows:
        new_row = np.array(row_data).reshape(1, -1)
        matrix = np.vstack([matrix, new_row])
    return matrix


def grow_array_dynamically(stream):
    collected = np.array([], dtype=np.float64)
    for value in stream:
        collected = np.append(collected, value)
    return collected
