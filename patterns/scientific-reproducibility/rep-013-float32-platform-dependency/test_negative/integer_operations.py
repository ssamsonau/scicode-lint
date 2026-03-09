import numpy as np


def count_occurrences(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))


def compute_histogram(data, bins=10):
    counts, _ = np.histogram(data, bins=bins)
    return counts


def argmax_index(scores):
    return np.argmax(scores)


def sort_indices(values):
    return np.argsort(values)


class IntegerMetrics:
    def __init__(self, predictions, labels):
        self.predictions = np.array(predictions, dtype=np.int64)
        self.labels = np.array(labels, dtype=np.int64)

    def accuracy(self):
        return np.mean(self.predictions == self.labels)

    def confusion_matrix(self):
        n_classes = max(self.labels.max(), self.predictions.max()) + 1
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for pred, label in zip(self.predictions, self.labels):
            cm[label, pred] += 1
        return cm
