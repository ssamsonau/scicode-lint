from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score


def cross_validate_timeseries(df, target_col, feature_cols):
    X = df[feature_cols].values
    y = df[target_col].values
    model = GradientBoostingRegressor()
    kf = KFold(n_splits=5, shuffle=True)
    scores = cross_val_score(model, X, y, cv=kf)
    return scores.mean()


def evaluate_model(df, features, target):
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    model = GradientBoostingRegressor()
    return cross_val_score(model, df[features], df[target], cv=cv)
