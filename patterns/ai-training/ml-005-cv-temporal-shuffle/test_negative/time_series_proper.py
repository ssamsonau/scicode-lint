from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


def cross_validate_timeseries(df, target_col, feature_cols):
    df = df.sort_values("date")
    X = df[feature_cols].values
    y = df[target_col].values
    model = GradientBoostingRegressor()
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=tscv)
    return scores.mean()


def evaluate_model(df, features, target, date_col="date"):
    df = df.sort_values(date_col)
    cv = TimeSeriesSplit(n_splits=10)
    model = GradientBoostingRegressor()
    return cross_val_score(model, df[features], df[target], cv=cv)
