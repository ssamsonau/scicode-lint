from collections.abc import Callable

import numpy as np


def create_expanding_features(values: np.ndarray, min_periods: int = 10) -> dict[str, np.ndarray]:
    """Create expanding window statistics using only historical data."""
    n = len(values)
    features = {
        "expanding_mean": np.full(n, np.nan),
        "expanding_std": np.full(n, np.nan),
        "expanding_max": np.full(n, np.nan),
    }

    for i in range(min_periods, n):
        window = values[:i]
        features["expanding_mean"][i] = np.mean(window)
        features["expanding_std"][i] = np.std(window)
        features["expanding_max"][i] = np.max(window)

    return features


class RollingFeatureBuilder:
    """Build rolling window features respecting temporal order."""

    def __init__(self, windows: list[int], aggregations: list[Callable]):
        self.windows = windows
        self.aggregations = aggregations

    def compute(self, series: np.ndarray) -> np.ndarray:
        n_features = len(self.windows) * len(self.aggregations)
        result = np.full((len(series), n_features), np.nan)

        col = 0
        for window in self.windows:
            for agg_func in self.aggregations:
                for i in range(window, len(series)):
                    result[i, col] = agg_func(series[i - window : i])
                col += 1

        return result
