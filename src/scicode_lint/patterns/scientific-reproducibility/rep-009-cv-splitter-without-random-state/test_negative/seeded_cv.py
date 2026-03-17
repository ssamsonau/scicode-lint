"""Cross-validation using sklearn's cross_val_score with consistent seeding."""

import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_validate


def evaluate_model_cv(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> dict:
    """Evaluate using cross_validate with default LeaveOneOut (no randomness)."""
    return cross_validate(
        estimator,
        X,
        y,
        cv=LeaveOneOut(),
        return_train_score=True,
    )


def compare_models_cv(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> dict[str, np.ndarray]:
    """Compare multiple models using cross_val_score with deterministic CV."""
    return {name: cross_val_score(model, X, y, cv=cv) for name, model in models.items()}


def time_series_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """Time series CV using TimeSeriesSplit (inherently deterministic)."""
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X))
