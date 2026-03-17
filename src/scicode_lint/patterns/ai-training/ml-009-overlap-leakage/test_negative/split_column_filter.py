import pandas as pd
from sklearn.linear_model import LogisticRegression


def prepare_data(data: pd.DataFrame):
    """Prepare train/test from pre-annotated split column."""
    train_data = data[data["split"] == "train"]
    test_data = data[data["split"] == "test"]

    return train_data, test_data


def load_and_train():
    data = pd.read_csv("full_dataset.csv")

    train, test = prepare_data(data)

    X_train = train.drop(["target", "split"], axis=1)
    y_train = train["target"]
    X_test = test.drop(["target", "split"], axis=1)
    y_test = test["target"]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, accuracy
