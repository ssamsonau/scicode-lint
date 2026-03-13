import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def evaluate_model(X, y):
    """Use CV score as final metric - no held-out test set."""
    model = SVC(kernel="rbf", C=1.0)

    # Cross-validation for model evaluation
    cv_scores = cross_val_score(model, X, y, cv=5)

    print(f"CV scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.2%}")

    # Issue: CV mean is reported as final performance
    # No separate held-out test set exists
    return np.mean(cv_scores)


def main():
    X = np.random.randn(500, 10)
    y = np.random.randint(0, 2, 500)

    final_score = evaluate_model(X, y)
    print(f"Final model performance: {final_score:.2%}")
