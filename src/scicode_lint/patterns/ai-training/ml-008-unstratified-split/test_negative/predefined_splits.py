"""Loading pre-defined splits from files."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_with_predefined_split():
    """Load pre-defined train/test split from files."""
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")

    X_train = train_data.drop("label", axis=1)
    y_train = train_data["label"]
    X_test = test_data.drop("label", axis=1)
    y_test = test_data["label"]

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def train_with_index_file():
    """Load split indices from numpy file."""
    train_idx = np.load("splits/train_indices.npy")
    test_idx = np.load("splits/test_indices.npy")

    X = np.load("data/features.npy")
    y = np.load("data/labels.npy")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)
