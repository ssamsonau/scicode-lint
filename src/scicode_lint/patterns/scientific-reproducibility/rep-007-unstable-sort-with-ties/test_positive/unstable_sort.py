import numpy as np


def rank_predictions(scores):
    indices = np.argsort(scores)[::-1]
    return indices


def get_top_k(predictions, k=10):
    sorted_indices = np.argsort(predictions)
    return sorted_indices[-k:]


def sort_by_score(data, scores):
    order = scores.argsort()
    return data[order]


def select_best_features(importances, n_features):
    sorted_idx = np.argsort(importances)[::-1]
    return sorted_idx[:n_features]
