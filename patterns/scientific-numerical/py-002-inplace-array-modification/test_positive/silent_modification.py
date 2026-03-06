import numpy as np


def normalize_weights(weights):
    total = weights.sum()
    weights /= total
    return weights


def remove_outliers(measurements):
    mean = measurements.mean()
    std = measurements.std()
    lower = mean - 3 * std
    upper = mean + 3 * std
    measurements[(measurements < lower) | (measurements > upper)] = mean
    return measurements


def augment_features(features):
    features[:, 0] *= 2
    features[:, 1] += 1
    return features


def apply_activation(outputs):
    outputs[outputs < 0] = 0
    return outputs


w = np.array([1.0, 2.0, 3.0])
normalized = normalize_weights(w)

data = np.random.rand(100, 3)
augmented = augment_features(data)
