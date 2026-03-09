import numpy as np


def normalize_features(X):
    feature_means = X.mean(axis=0)
    centered = X - feature_means

    feature_std = X.std(axis=0)
    normalized = centered / feature_std
    return normalized


def batch_normalize(batch):
    batch_mean = batch.mean(axis=1)
    batch_std = batch.std(axis=1)

    normalized = (batch - batch_mean) / batch_std
    return normalized


X = np.random.rand(100, 50)  # (samples, features)
X_norm = normalize_features(X)

batch = np.random.rand(32, 128)
batch_norm = batch_normalize(batch)
