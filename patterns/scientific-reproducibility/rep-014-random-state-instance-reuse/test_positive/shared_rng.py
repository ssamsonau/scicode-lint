import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


def train_models(X, y):
    rng = np.random.RandomState(42)

    rf = RandomForestClassifier(random_state=rng)
    gb = GradientBoostingClassifier(random_state=rng)

    rf.fit(X, y)
    gb.fit(X, y)

    return rf, gb


def cross_validate_ensemble(X, y, n_estimators=5):
    rng = np.random.RandomState(0)
    estimators = []
    for _ in range(n_estimators):
        est = RandomForestClassifier(random_state=rng)
        estimators.append(est)
    return estimators
