import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC


def nested_cv_evaluation(X, y):
    """Nested CV without held-out test set - still multi-test leakage.

    Even with nested CV (inner CV for tuning, outer CV for evaluation),
    if there's no separate held-out test set, the outer CV scores are
    still optimistically biased because the entire dataset was used.
    """
    param_grid = {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}

    # Inner CV for hyperparameter tuning
    inner_cv = GridSearchCV(SVC(), param_grid, cv=3)

    # Outer CV for "evaluation" - but this is on the same data!
    outer_scores = cross_val_score(inner_cv, X, y, cv=5)

    print(f"Nested CV scores: {outer_scores}")
    print(f"Mean accuracy: {np.mean(outer_scores):.2%}")

    # Reporting nested CV mean as final performance - no held-out test
    return np.mean(outer_scores)


def main():
    X = np.random.randn(500, 15)
    y = np.random.randint(0, 2, 500)

    final_score = nested_cv_evaluation(X, y)
    print(f"Model achieves {final_score:.2%} accuracy")
