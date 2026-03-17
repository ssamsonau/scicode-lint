import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


class StratifiedDataSplitter:
    """Handles stratified splitting for imbalanced classification tasks."""

    def __init__(self, n_splits: int = 5, test_size: float = 0.2):
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.holdout = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    def get_train_test(self, X: np.ndarray, y: np.ndarray):
        train_idx, test_idx = next(self.holdout.split(X, y))
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def cross_validate(self, X: np.ndarray, y: np.ndarray):
        for train_idx, val_idx in self.cv.split(X, y):
            yield X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def prepare_multiclass_split(X, y, categorical_target=True):
    """Stratified split handling categorical and numerical targets."""
    if categorical_target:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = y

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    train_idx, test_idx = next(sss.split(X, y_encoded))

    return {
        "X_train": X[train_idx],
        "X_test": X[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx],
    }
