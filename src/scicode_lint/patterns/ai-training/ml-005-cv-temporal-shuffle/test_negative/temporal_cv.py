import numpy as np
from sklearn.linear_model import Ridge


class ExpandingWindowEvaluator:
    def __init__(self, min_train_size=100, step_size=50):
        self.min_train_size = min_train_size
        self.step_size = step_size

    def evaluate(self, X, y):
        n = len(X)
        scores = []
        model = Ridge(alpha=1.0)

        for end in range(self.min_train_size, n - self.step_size, self.step_size):
            X_train, y_train = X[:end], y[:end]
            X_val, y_val = X[end : end + self.step_size], y[end : end + self.step_size]

            model.fit(X_train, y_train)
            scores.append(model.score(X_val, y_val))

        return np.array(scores)


def walk_forward_backtest(prices, features, window=252):
    from sklearn.ensemble import GradientBoostingRegressor

    predictions = []
    for i in range(window, len(prices) - 1):
        X_train = features[i - window : i]
        y_train = prices[i - window : i]
        X_pred = features[i : i + 1]

        model = GradientBoostingRegressor(n_estimators=50, max_depth=3)
        model.fit(X_train, y_train)
        predictions.append(model.predict(X_pred)[0])

    return np.array(predictions)
