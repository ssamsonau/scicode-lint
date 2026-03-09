import numpy as np


def get_ranking(scores):
    return np.argsort(scores)


def top_k_indices(values, k):
    indices = np.argsort(values)[::-1]
    return indices[:k]


def order_by_priority(items, priorities):
    order = np.argsort(priorities)
    return [items[i] for i in order]
