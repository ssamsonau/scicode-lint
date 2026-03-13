import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


def train_from_separate_files():
    """Train and test loaded from completely separate files - no overlap possible."""
    # Separate data sources guarantee no overlap
    train_data = pd.read_csv("train_dataset.csv")
    test_data = pd.read_csv("test_dataset.csv")

    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]
    X_test = test_data.drop("target", axis=1)
    y_test = test_data["target"]

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy}")

    return model
