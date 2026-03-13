import numpy as np


def select_features_from_dict(X, feature_dict):
    selected = []
    for feature_name in feature_dict:
        idx = feature_dict[feature_name]
        selected.append(X[:, idx])
    return np.column_stack(selected)


def process_config(config):
    results = {}
    for key in config:
        results[key] = config[key] * 2
    return results


def iterate_ordered_dict(data):
    yield from data.items()


class FeatureSelector:
    def __init__(self, features):
        self.features = list(features.items())

    def select(self, X):
        return np.column_stack([X[:, idx] for _, idx in self.features])
