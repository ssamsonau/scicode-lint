"""Bagging ensemble with shared RandomState across tree classifiers."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def create_bagging_ensemble(X, y, n_estimators=10):
    rng = np.random.RandomState(123)
    estimators = []
    for _ in range(n_estimators):
        indices = rng.choice(len(X), size=len(X), replace=True)
        clf = DecisionTreeClassifier(random_state=rng)
        clf.fit(X[indices], y[indices])
        estimators.append(clf)
    return estimators
