import pandas as pd


class TemporalFeatureEngineer:
    """Create lagged features using only past data - no future leakage."""

    def __init__(self, lag_periods: list[int]):
        self.lag_periods = lag_periods

    def transform(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        result = df.copy()
        for lag in self.lag_periods:
            result[f"{target}_lag_{lag}"] = df[target].shift(lag)
        return result.dropna()


def build_autoregressive_features(series: pd.Series, max_lag: int = 5):
    """Build AR features using historical values only."""
    features = pd.DataFrame(index=series.index)
    for i in range(1, max_lag + 1):
        features[f"ar_{i}"] = series.shift(i)
    features["rolling_mean_3"] = series.shift(1).rolling(window=3).mean()
    features["rolling_std_5"] = series.shift(1).rolling(window=5).std()
    return features.dropna()
