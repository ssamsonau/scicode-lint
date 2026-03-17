import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedKFold


class ImbalancedClassifierTrainer:
    """Trainer for imbalanced classification without stratification."""

    def __init__(self, n_splits: int = 5, n_repeats: int = 3):
        self.cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
        self.model = GradientBoostingClassifier(n_estimators=50)

    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        scores = []
        for train_idx, test_idx in self.cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self.model.fit(X_train, y_train)
            scores.append(self.model.score(X_test, y_test))

        return scores


def bootstrap_evaluation(X, y, n_bootstrap: int = 100, sample_ratio: float = 0.8):
    """Bootstrap sampling without stratification for imbalanced data."""
    rng = np.random.default_rng(seed=42)
    n_samples = int(len(X) * sample_ratio)
    results = []

    for _ in range(n_bootstrap):
        indices = rng.choice(len(X), size=n_samples, replace=True)
        X_boot, y_boot = X[indices], y[indices]

        test_mask = np.ones(len(X), dtype=bool)
        test_mask[indices] = False
        X_test, y_test = X[test_mask], y[test_mask]

        if len(np.unique(y_boot)) < 2:
            continue

        model = GradientBoostingClassifier(n_estimators=20)
        model.fit(X_boot, y_boot)
        results.append(model.score(X_test, y_test))

    return np.mean(results), np.std(results)
