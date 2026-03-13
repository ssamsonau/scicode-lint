from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold


def get_reproducible_cv(n_splits=5, shuffle=True, random_state=42):
    """KFold with random_state for reproducible folds."""
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def stratified_cv_seeded(n_splits=5, random_state=42):
    """StratifiedKFold with fixed random state."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def create_shuffle_split(n_splits=10, test_size=0.2, random_state=42):
    """ShuffleSplit with reproducible random state."""
    return ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)


def cross_validate_reproducibly(model, X, y, n_splits=5, seed=42):
    """Cross-validation with seeded CV splitter."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in cv.split(X):
        model.fit(X[train_idx], y[train_idx])
        scores.append(model.score(X[val_idx], y[val_idx]))
    return scores
