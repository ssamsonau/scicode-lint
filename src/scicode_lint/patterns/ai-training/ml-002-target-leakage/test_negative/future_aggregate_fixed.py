import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit


def backtest_energy_model(consumption_df):
    consumption_df = consumption_df.sort_values("timestamp")
    consumption_df["hour"] = pd.to_datetime(consumption_df["timestamp"]).dt.hour
    consumption_df["dayofweek"] = pd.to_datetime(consumption_df["timestamp"]).dt.dayofweek
    consumption_df["lag_1h"] = consumption_df["kwh"].shift(1)
    consumption_df["lag_24h"] = consumption_df["kwh"].shift(24)
    consumption_df = consumption_df.dropna()

    feature_cols = ["hour", "dayofweek", "temperature", "lag_1h", "lag_24h"]
    X = consumption_df[feature_cols]
    y = consumption_df["kwh"]

    tscv = TimeSeriesSplit(n_splits=4)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        model = GradientBoostingRegressor(n_estimators=100)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        scores.append(model.score(X.iloc[test_idx], y.iloc[test_idx]))
    return scores
