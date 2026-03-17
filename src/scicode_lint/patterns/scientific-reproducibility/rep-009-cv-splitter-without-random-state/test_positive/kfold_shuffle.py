"""Hyperparameter search with nested CV missing random_state."""

from sklearn.model_selection import KFold, ShuffleSplit


class NestedCVSearch:
    """Grid search with nested cross-validation - non-reproducible splits."""

    def __init__(self, param_grid: dict):
        self.param_grid = param_grid
        self.outer_cv = KFold(n_splits=5, shuffle=True)
        self.inner_cv = ShuffleSplit(n_splits=3, test_size=0.2)

    def fit(self, X, y, model_class):
        outer_scores = []
        for train_idx, test_idx in self.outer_cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            best_score = -1
            best_params = None
            for params in self._param_combinations():
                inner_scores = []
                for tr, val in self.inner_cv.split(X_train):
                    model = model_class(**params)
                    model.fit(X_train[tr], y_train[tr])
                    inner_scores.append(model.score(X_train[val], y_train[val]))
                avg = sum(inner_scores) / len(inner_scores)
                if avg > best_score:
                    best_score = avg
                    best_params = params

            final_model = model_class(**best_params)
            final_model.fit(X_train, y_train)
            outer_scores.append(final_model.score(X_test, y_test))
        return outer_scores

    def _param_combinations(self):
        keys = list(self.param_grid.keys())
        if len(keys) == 1:
            return [{keys[0]: v} for v in self.param_grid[keys[0]]]
        return [dict(zip(keys, vals)) for vals in zip(*self.param_grid.values())]
