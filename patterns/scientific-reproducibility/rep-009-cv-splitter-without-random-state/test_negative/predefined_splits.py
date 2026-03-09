import numpy as np
from sklearn.model_selection import PredefinedSplit


def use_predefined_splits(X, y, model, fold_assignments):
    ps = PredefinedSplit(fold_assignments)

    scores = []
    for train_idx, val_idx in ps.split():
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        scores.append(model.score(X_val, y_val))

    return scores


def train_val_split_fixed(X, y, model, val_fold=0):
    n_samples = len(X)
    test_fold = np.array([val_fold if i < n_samples // 5 else -1 for i in range(n_samples)])
    ps = PredefinedSplit(test_fold)

    for train_idx, val_idx in ps.split():
        model.fit(X[train_idx], y[train_idx])
        return model.score(X[val_idx], y[val_idx])


class FixedSplitCV:
    def __init__(self, splits_file):
        self.splits = np.load(splits_file)

    def get_splits(self, X):
        ps = PredefinedSplit(self.splits)
        return list(ps.split())
