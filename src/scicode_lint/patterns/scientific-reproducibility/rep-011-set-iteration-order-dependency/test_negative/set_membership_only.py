import numpy as np


def filter_valid_features(X, valid_set, feature_indices):
    selected = []
    for idx in feature_indices:
        if idx in valid_set:
            selected.append(X[:, idx])
    return np.column_stack(selected) if selected else np.array([])


def deduplicate_preserving_order(items):
    seen = set()
    results = []
    for item in items:
        if item not in seen:
            seen.add(item)
            results.append(item)
    return results


def count_unique(items):
    unique = set(items)
    return len(unique)


def has_duplicates(items):
    seen = set()
    for item in items:
        if item in seen:
            return True
        seen.add(item)
    return False


def intersect_features(features_a, features_b, ordered_list):
    valid = features_a & features_b
    return [f for f in ordered_list if f in valid]
