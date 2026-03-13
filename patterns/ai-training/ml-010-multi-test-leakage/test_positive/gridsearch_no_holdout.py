import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


def tune_and_report(X, y):
    """GridSearchCV best score as final result - no held-out test."""
    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "learning_rate_init": [0.001, 0.01],
    }

    model = MLPClassifier(max_iter=200, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X, y)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.2%}")

    # Issue: GridSearchCV best_score_ is the final reported metric
    # This is the validation score from CV, not a held-out test
    return grid_search.best_score_


def main():
    X = np.random.randn(800, 15)
    y = np.random.randint(0, 2, 800)

    final_accuracy = tune_and_report(X, y)
    print(f"Model achieves {final_accuracy:.2%} accuracy")
