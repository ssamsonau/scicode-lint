"""Data preprocessing pipeline reusing RandomState across operations."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer


def preprocess_pipeline(X, y):
    rng = np.random.RandomState(42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rng
    )

    qt = QuantileTransformer(random_state=rng, output_distribution="normal")
    X_train = qt.fit_transform(X_train)
    X_test = qt.transform(X_test)

    pca = PCA(n_components=10)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    noise = rng.randn(*X_train.shape) * 0.01
    X_train = X_train + noise

    return X_train, X_test, y_train, y_test
