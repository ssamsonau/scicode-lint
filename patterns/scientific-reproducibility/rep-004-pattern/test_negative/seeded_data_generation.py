import numpy as np
import torch

RANDOM_SEED = 2024

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def generate_synthetic_data(n_samples):
    features = np.random.randn(n_samples, 50)
    noise = np.random.uniform(-0.5, 0.5, n_samples)
    targets = features.sum(axis=1) + noise
    return features, targets


def split_dataset(X, y, test_ratio):
    n = len(X)
    indices = np.random.permutation(n)
    split_point = int(n * (1 - test_ratio))

    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


X, y = generate_synthetic_data(1000)
X_train, y_train, X_test, y_test = split_dataset(X, y, 0.2)

model = torch.nn.Linear(50, 1)
