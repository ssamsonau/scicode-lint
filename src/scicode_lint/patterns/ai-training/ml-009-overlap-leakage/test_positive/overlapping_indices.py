import numpy as np
from sklearn.linear_model import LogisticRegression


def split_data_wrong(data, labels):
    train = data[:800]
    train_labels = labels[:800]

    test = data[700:]
    test_labels = labels[700:]

    return train, test, train_labels, test_labels


def train_and_evaluate():
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)

    X_train, X_test, y_train, y_test = split_data_wrong(X, y)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy}")

    return model
