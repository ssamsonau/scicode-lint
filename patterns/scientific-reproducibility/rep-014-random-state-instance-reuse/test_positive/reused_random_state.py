import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_ensemble(X, y):
    rng = np.random.RandomState(42)

    rf = RandomForestClassifier(n_estimators=100, random_state=rng)
    lr = LogisticRegression(random_state=rng)

    rf.fit(X, y)
    lr.fit(X, y)

    return rf, lr


def split_and_train(X, y):
    rng = np.random.RandomState(42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rng)

    model = RandomForestClassifier(random_state=rng)
    model.fit(X_train, y_train)

    return model
