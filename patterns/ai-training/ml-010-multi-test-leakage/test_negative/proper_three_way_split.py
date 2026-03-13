import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def train_with_three_splits(X, y):
    """Correct: three-way split with held-out test set."""
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Second split: train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )

    best_model = None
    best_val_acc = 0

    # Tune on validation set
    for n_estimators in [50, 100, 200]:
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        val_acc = model.score(X_val, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model

    # Final evaluation on held-out test set (never used during tuning)
    test_acc = best_model.score(X_test, y_test)
    print(f"Validation accuracy: {best_val_acc:.2%}")
    print(f"Final test accuracy: {test_acc:.2%}")

    return best_model, test_acc


def main():
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 2, 1000)

    model, accuracy = train_with_three_splits(X, y)
