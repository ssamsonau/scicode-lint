import pickle

import joblib
from sklearn.ensemble import RandomForestClassifier


def save_model(model, filepath):
    joblib.dump(model, filepath)


def save_model_pickle(model, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def train_and_save(X, y):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    save_model(model, "model.joblib")


if __name__ == "__main__":
    import numpy as np

    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    train_and_save(X, y)
