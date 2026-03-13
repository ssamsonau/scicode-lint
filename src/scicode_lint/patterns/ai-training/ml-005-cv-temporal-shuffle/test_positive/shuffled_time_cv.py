from sklearn.model_selection import KFold


def time_series_cv(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        scores.append(train_idx.mean())
    return scores


def validate_temporal(df, target_col):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    X = df.drop(columns=[target_col, "date"])
    y = df[target_col]
    model = RandomForestRegressor()
    return cross_val_score(model, X, y, cv=5)
