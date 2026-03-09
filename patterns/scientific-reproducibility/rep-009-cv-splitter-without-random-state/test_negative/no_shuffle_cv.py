from sklearn.model_selection import KFold


def cross_validate_ordered(X, y, model, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=False)

    scores = []
    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        scores.append(model.score(X_val, y_val))

    return scores


def time_series_cv(X, y, model, n_splits=5):
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        model.fit(X[train_idx], y[train_idx])
        scores.append(model.score(X[val_idx], y[val_idx]))
    return scores


def leave_one_out(X, y, model):
    from sklearn.model_selection import LeaveOneOut

    loo = LeaveOneOut()
    scores = []
    for train_idx, val_idx in loo.split(X):
        model.fit(X[train_idx], y[train_idx])
        scores.append(model.score(X[val_idx], y[val_idx]))
    return scores
