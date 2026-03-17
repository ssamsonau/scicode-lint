import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def split_with_different_seeds(data: pd.DataFrame):
    """Split using .sample() with different seeds - unlikely but possible overlap."""
    train = data.sample(frac=0.8, random_state=42)
    test = data.sample(frac=0.2, random_state=99)

    return train, test


def train_model():
    data = pd.read_csv("dataset.csv")

    train, test = split_with_different_seeds(data)

    X_train = train.drop("target", axis=1)
    y_train = train["target"]
    X_test = test.drop("target", axis=1)
    y_test = test["target"]

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, accuracy
