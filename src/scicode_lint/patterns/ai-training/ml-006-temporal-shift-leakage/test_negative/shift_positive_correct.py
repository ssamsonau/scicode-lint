import numpy as np


class SequenceDatasetBuilder:
    def __init__(self, lookback=30, horizon=1):
        self.lookback = lookback
        self.horizon = horizon

    def build_sequences(self, time_series):
        X, y = [], []
        for i in range(self.lookback, len(time_series) - self.horizon + 1):
            X.append(time_series[i - self.lookback : i])
            y.append(time_series[i + self.horizon - 1])
        return np.array(X), np.array(y)


def create_return_features(prices):
    log_returns = np.diff(np.log(prices))
    volatility = np.array(
        [np.std(log_returns[max(0, i - 20) : i]) for i in range(1, len(log_returns) + 1)]
    )
    momentum = np.array(
        [np.sum(log_returns[max(0, i - 5) : i]) for i in range(1, len(log_returns) + 1)]
    )
    return np.column_stack([log_returns, volatility, momentum])
