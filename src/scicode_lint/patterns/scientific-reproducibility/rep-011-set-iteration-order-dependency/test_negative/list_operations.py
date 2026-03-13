import numpy as np


def select_features(X, feature_list):
    selected = []
    for feature in feature_list:
        selected.append(X[:, feature])
    return np.column_stack(selected)


def get_unique_labels(labels):
    seen = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    return seen


def process_ordered_items(items):
    results = []
    for item in items:
        results.append(process(item))
    return results


def process(item):
    return item * 2
