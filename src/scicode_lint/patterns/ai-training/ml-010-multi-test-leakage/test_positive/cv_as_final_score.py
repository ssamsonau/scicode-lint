"""Cross-validation for model evaluation."""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def evaluate_model(X, y):
    """Evaluate model using cross-validation."""
    model = SVC(kernel="rbf", C=1.0)

    cv_scores = cross_val_score(model, X, y, cv=5)

    print(f"CV scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.2%}")

    return np.mean(cv_scores)


def main():
    X = np.random.randn(500, 10)
    y = np.random.randint(0, 2, 500)

    final_score = evaluate_model(X, y)
    print(f"Final model performance: {final_score:.2%}")
