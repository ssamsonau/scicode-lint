import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_properly(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, accuracy


def main():
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)

    model, acc = train_properly(X, y)
    print(f"Test accuracy: {acc}")
