"""Train and evaluate classifier on dataset."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_and_evaluate(X, y):
    """Train classifier and report performance."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model = None
    best_val_acc = 0

    for n_estimators in [10, 50, 100]:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        val_acc = model.score(X_val, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model

    print(f"Final accuracy: {best_val_acc}")
    return best_model, best_val_acc


def main():
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 2, 1000)

    model, accuracy = train_and_evaluate(X, y)
    print(f"Model achieved {accuracy:.2%} accuracy")
