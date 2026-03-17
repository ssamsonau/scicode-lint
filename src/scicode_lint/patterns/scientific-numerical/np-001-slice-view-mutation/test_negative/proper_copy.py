import numpy as np


def process_subset_with_copy(data):
    subset = data[10:50].copy()
    subset += 5
    return subset


def normalize_column(matrix, col_idx):
    column = matrix[:, col_idx].copy()
    column -= column.mean()
    column /= column.std()
    return column


def extract_region(image, x1, x2, y1, y2):
    region = image[x1:x2, y1:y2].copy()
    region *= 255
    return region


def process_batch(data, batch_size):
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size].copy()
        batch -= batch.min()
        results.append(batch)
    return results


arr = np.random.rand(100)
subset = process_subset_with_copy(arr)
