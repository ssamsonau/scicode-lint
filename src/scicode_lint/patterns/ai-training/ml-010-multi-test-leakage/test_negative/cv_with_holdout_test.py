"""CV for tuning with separate held-out test set."""

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


def tune_with_cv_and_test(X, y):
    """Tune with CV and evaluate on held-out test set."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}

    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)

    print(f"Best CV score (on train): {grid_search.best_score_:.2%}")
    print(f"Best params: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    test_acc = best_model.score(X_test, y_test)
    print(f"Final test accuracy: {test_acc:.2%}")

    return best_model, test_acc


def main():
    X = np.random.randn(800, 15)
    y = np.random.randint(0, 2, 800)

    model, accuracy = tune_with_cv_and_test(X, y)
