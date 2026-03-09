import numpy as np


def process_data(data):
    temp1 = data * 2
    temp2 = temp1 + 10
    temp3 = np.sqrt(temp2)
    temp4 = temp3 / temp3.max()
    return temp4


def feature_engineering(X):
    centered = X - X.mean(axis=0)
    squared = centered**2
    summed = squared.sum(axis=1)
    result = np.sqrt(summed)
    return result
