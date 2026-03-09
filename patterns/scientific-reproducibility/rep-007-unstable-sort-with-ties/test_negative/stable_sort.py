import numpy as np


def rank_with_stable_sort(scores):
    """Using kind='stable' for reproducible tie-breaking."""
    return np.argsort(scores, kind="stable")


def get_top_k_stable(values, k=10):
    """Stable sort for selecting top-k with consistent tie handling."""
    indices = np.argsort(-values, kind="stable")
    return indices[:k]


def rank_features(importances):
    """Feature ranking with stable sort for reproducible results."""
    ranking = np.argsort(-importances, kind="stable")
    return ranking


def sort_by_score_stable(items, scores):
    """Sort items by scores using stable sort."""
    order = np.argsort(scores, kind="stable")
    return [items[i] for i in order]
