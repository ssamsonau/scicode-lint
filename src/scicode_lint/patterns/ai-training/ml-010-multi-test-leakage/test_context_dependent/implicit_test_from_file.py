"""Training for competition with external test set."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_for_competition(train_file: str):
    """Train model for competition submission."""
    data = pd.read_csv(train_file)
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model = None
    best_val_acc = 0

    for n_estimators in [50, 100, 200]:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        val_acc = model.score(X_val, y_val)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model

    print(f"Local validation accuracy: {best_val_acc:.2%}")

    return best_model


def make_submission(model, test_file: str, output_file: str):
    """Generate predictions for external test set."""
    test_data = pd.read_csv(test_file)
    predictions = model.predict(test_data)

    submission = pd.DataFrame({"id": test_data["id"], "target": predictions})
    submission.to_csv(output_file, index=False)
