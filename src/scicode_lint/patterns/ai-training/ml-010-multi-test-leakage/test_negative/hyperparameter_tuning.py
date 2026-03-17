"""Hyperparameter tuning script."""

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


def tune_hyperparameters(X, y):
    """Find best hyperparameters using cross-validation."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        cv=5,
        scoring="accuracy",
    )
    grid_search.fit(X_train, y_train)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_}")

    val_score = grid_search.score(X_val, y_val)
    print(f"Validation score: {val_score}")

    return grid_search.best_estimator_


if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, random_state=42)
    best_model = tune_hyperparameters(X, y)
