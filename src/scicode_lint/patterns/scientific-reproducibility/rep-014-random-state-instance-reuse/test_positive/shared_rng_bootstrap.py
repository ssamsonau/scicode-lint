import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


def bootstrap_samples(X, y, n_bootstrap=100):
    rng = np.random.RandomState(42)
    samples = []
    for _ in range(n_bootstrap):
        X_boot, y_boot = resample(X, y, random_state=rng)
        samples.append((X_boot, y_boot))
    return samples


def create_bagging_ensemble(n_estimators=10):
    rng = np.random.RandomState(123)
    estimators = []
    for _ in range(n_estimators):
        clf = DecisionTreeClassifier(random_state=rng)
        estimators.append(clf)
    return estimators
