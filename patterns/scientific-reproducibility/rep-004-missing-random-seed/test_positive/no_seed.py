import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def run_experiment(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def generate_synthetic_data(n_samples=1000):
    X = np.random.randn(n_samples, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y
