import numpy as np


def normalize_data(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


def scale_features(X, min_val=0, max_val=1):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (X - X_min) / (X_max - X_min) * (max_val - min_val) + min_val


def compute_gradients(loss, params):
    return np.gradient(loss)


class DataProcessor:
    def __init__(self, data):
        self.data = data.astype(np.float32)

    def transform(self):
        centered = self.data - self.data.mean(axis=0)
        scaled = centered / centered.std(axis=0)
        return scaled

    def pca(self, n_components):
        centered = self.data - self.data.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = eigenvalues.argsort()[::-1]
        return eigenvectors[:, idx[:n_components]]
