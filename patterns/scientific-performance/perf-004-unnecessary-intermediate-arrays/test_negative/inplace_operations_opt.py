import numpy as np


def process_data(data):
    result = data * 2
    result += 10
    np.sqrt(result, out=result)
    result /= result.max()
    return result


def feature_engineering(X):
    centered = X - X.mean(axis=0)
    return np.sqrt(np.sum(centered**2, axis=1))
