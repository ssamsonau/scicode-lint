import numpy as np


def process_data(data):
    result = data * 2
    result += 10
    np.sqrt(result, out=result)
    result /= result.max()
    return result


def feature_engineering(X):
    return np.linalg.norm(X - X.mean(axis=0), axis=1)
