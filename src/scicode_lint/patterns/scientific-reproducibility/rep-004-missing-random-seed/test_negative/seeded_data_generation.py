"""Data generation using sklearn's make_* utilities with random_state."""

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


def create_classification_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    random_state: int = 42,
):
    """Generate synthetic classification data with reproducible randomness."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        random_state=random_state,
    )
    return train_test_split(X, y, test_size=0.2, random_state=random_state)


def create_regression_dataset(
    n_samples: int = 1000,
    n_features: int = 20,
    random_state: int = 42,
):
    """Generate synthetic regression data with reproducible randomness."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        noise=0.1,
        random_state=random_state,
    )
    return X, y
