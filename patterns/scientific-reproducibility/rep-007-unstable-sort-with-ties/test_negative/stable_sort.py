import numpy as np


def rank_with_stable_sort(scores):
    return np.argsort(scores, kind="stable")


def get_top_k_stable(values, k=10):
    indices = np.argsort(-values, kind="stable")
    return indices[:k]


def rank_features(importances):
    ranking = np.argsort(-importances, kind="stable")
    return ranking


def sort_by_score_stable(items, scores):
    order = np.argsort(scores, kind="stable")
    return [items[i] for i in order]
