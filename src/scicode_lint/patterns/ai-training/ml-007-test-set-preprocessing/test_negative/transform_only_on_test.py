import numpy as np


class ZScoreNormalizer:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_


def normalize_train_test(train_features, test_features):
    normalizer = ZScoreNormalizer()
    normalizer.fit(train_features)
    train_normed = normalizer.transform(train_features)
    test_normed = normalizer.transform(test_features)
    return train_normed, test_normed


def clip_outliers_from_train_stats(train_data, test_data, n_sigma=3):
    mu = np.mean(train_data, axis=0)
    sigma = np.std(train_data, axis=0) + 1e-8
    lower = mu - n_sigma * sigma
    upper = mu + n_sigma * sigma
    train_clipped = np.clip(train_data, lower, upper)
    test_clipped = np.clip(test_data, lower, upper)
    return train_clipped, test_clipped
