from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold


def cross_validate_model(model, X, y, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True)
    scores = []
    for train_idx, val_idx in kfold.split(X):
        model.fit(X[train_idx], y[train_idx])
        scores.append(model.score(X[val_idx], y[val_idx]))
    return scores


def stratified_cv(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for train, test in skf.split(X, y):
        model.fit(X[train], y[train])


def shuffle_split_validation(X, y):
    ss = ShuffleSplit(n_splits=10, test_size=0.2)
    for train_index, test_index in ss.split(X):
        yield X[train_index], X[test_index]
