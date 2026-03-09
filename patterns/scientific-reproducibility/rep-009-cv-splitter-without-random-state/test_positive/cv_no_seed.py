from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    ShuffleSplit,
    StratifiedKFold,
)


def cross_validate_model(X, y, model):
    kfold = KFold(n_splits=5, shuffle=True)

    scores = []
    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        scores.append(model.score(X_val, y_val))

    return scores


def stratified_cv(X, y, model):
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    return [model.fit(X[t], y[t]).score(X[v], y[v]) for t, v in skf.split(X, y)]


def repeated_cv(X, y, model):
    rkf = RepeatedKFold(n_splits=5, n_repeats=3)
    return list(rkf.split(X))


def shuffle_split_cv(X, y):
    ss = ShuffleSplit(n_splits=5, test_size=0.2)
    return list(ss.split(X))
