import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_and_evaluate(X, y):
    """Train with only train/val split - no held-out test set.

    The X_val naming is correct (this IS validation data used for tuning).
    The issue is that val_acc is reported as final performance with no
    separate held-out test set for unbiased evaluation.
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model = None
    best_val_acc = 0

    # Tune on validation
    for n_estimators in [10, 50, 100]:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        val_acc = model.score(X_val, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model

    # Issue: Reporting validation accuracy as final result
    # No separate test set exists!
    print(f"Final accuracy: {best_val_acc}")
    return best_model, best_val_acc


def main():
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 2, 1000)

    model, accuracy = train_and_evaluate(X, y)
    print(f"Model achieved {accuracy:.2%} accuracy")
