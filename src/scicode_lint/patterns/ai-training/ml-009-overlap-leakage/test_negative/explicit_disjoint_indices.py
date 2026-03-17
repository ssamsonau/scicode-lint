import numpy as np
from sklearn.svm import SVC


def split_with_explicit_indices(X, y):
    """Split using explicit disjoint index ranges - no overlap."""
    n_samples = len(X)
    split_idx = int(0.8 * n_samples)

    train_idx = np.arange(0, split_idx)
    test_idx = np.arange(split_idx, n_samples)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, X_test, y_train, y_test


def train_model():
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 2, 1000)

    X_train, X_test, y_train, y_test = split_with_explicit_indices(X, y)

    model = SVC()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, accuracy
