import numpy as np


def normalize_weights(weights):
    total = weights.sum()
    normalized = weights / total
    return normalized


def remove_outliers(measurements):
    mean = measurements.mean()
    std = measurements.std()
    lower = mean - 3 * std
    upper = mean + 3 * std
    mask = (measurements >= lower) & (measurements <= upper)
    cleaned = np.where(mask, measurements, mean)
    return cleaned


def augment_features(features):
    augmented = features.copy()
    augmented[:, 0] = features[:, 0] * 2
    augmented[:, 1] = features[:, 1] + 1
    return augmented


def apply_activation(outputs):
    activated = np.maximum(outputs, 0)
    return activated


w = np.array([1.0, 2.0, 3.0])
normalized = normalize_weights(w)

data = np.random.rand(100, 3)
augmented = augment_features(data)
