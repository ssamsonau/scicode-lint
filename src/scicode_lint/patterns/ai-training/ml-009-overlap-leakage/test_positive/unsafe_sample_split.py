import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def split_with_sample(data: pd.DataFrame):
    """Split data using .sample() - may have overlapping rows."""
    train = data.sample(frac=0.8, random_state=42)
    test = data.sample(frac=0.2, random_state=123)

    return train, test


def train_model():
    data = pd.read_csv("dataset.csv")

    train, test = split_with_sample(data)

    X_train, y_train = train.drop("target", axis=1), train["target"]
    X_test, y_test = test.drop("target", axis=1), test["target"]

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy}")

    return model
