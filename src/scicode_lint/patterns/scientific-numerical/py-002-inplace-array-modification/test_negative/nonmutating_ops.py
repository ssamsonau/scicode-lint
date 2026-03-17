import numpy as np


def softmax(logits):
    shifted = logits - logits.max()
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()


def one_hot_encode(labels, num_classes):
    n = len(labels)
    encoded = np.zeros((n, num_classes))
    encoded[np.arange(n), labels] = 1.0
    return encoded


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b + 1e-8)


def batch_zscore(matrix):
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0) + 1e-8
    return (matrix - means) / stds


def moving_window_variance(data, window):
    n = len(data)
    result = np.zeros(n - window + 1)
    for i in range(len(result)):
        result[i] = data[i : i + window].var()
    return result


logits = np.array([2.0, 1.0, 0.5, -1.0])
probs = softmax(logits)

features = np.random.randn(50, 8)
normalized = batch_zscore(features)
