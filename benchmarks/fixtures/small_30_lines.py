"""Small scientific script: basic data normalization pipeline (~30 lines)."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_normalize(filepath):
    data = np.loadtxt(filepath, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def compute_statistics(arr):
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    normalized = (arr - mean) / std
    return normalized, mean, std


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_normalize("data.csv")
    X_norm, mu, sigma = compute_statistics(X_train)
    print(f"Train shape: {X_norm.shape}, mean: {mu.mean():.4f}")
