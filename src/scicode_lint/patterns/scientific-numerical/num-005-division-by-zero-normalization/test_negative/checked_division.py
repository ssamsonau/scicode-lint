import numpy as np


def percentage_of_total(values):
    total = np.sum(values)
    if total == 0:
        return np.zeros_like(values)
    return 100.0 * values / total


def probability_distribution(counts):
    total = counts.sum()
    if total == 0:
        return np.ones(len(counts)) / len(counts)
    return counts / total


def weighted_average(values, weights):
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        return np.mean(values)
    return np.sum(values * weights) / weight_sum


def divide_with_where(a, b):
    return np.where(b != 0, a / b, 0.0)


data = np.array([10, 20, 30, 40])
pcts = percentage_of_total(data)
