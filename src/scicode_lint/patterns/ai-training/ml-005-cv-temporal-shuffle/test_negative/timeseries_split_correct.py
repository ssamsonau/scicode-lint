import numpy as np
from sklearn.ensemble import RandomForestRegressor


class PurgedTimeSeriesCV:
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None):
        n = len(X)
        embargo = int(n * self.embargo_pct)
        fold_size = n // (self.n_splits + 1)

        for i in range(self.n_splits):
            test_start = (i + 1) * fold_size
            test_end = test_start + fold_size
            train_end = test_start - embargo

            train_idx = np.arange(0, max(0, train_end))
            test_idx = np.arange(test_start, min(test_end, n))
            yield train_idx, test_idx


def evaluate_with_purged_cv(X, y, n_splits=5):
    cv = PurgedTimeSeriesCV(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        model = RandomForestRegressor(n_estimators=50, random_state=0)
        model.fit(X[train_idx], y[train_idx])
        scores.append(model.score(X[test_idx], y[test_idx]))
    return np.array(scores)
