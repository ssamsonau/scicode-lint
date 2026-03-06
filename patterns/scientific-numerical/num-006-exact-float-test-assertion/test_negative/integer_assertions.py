import numpy as np


def test_array_shape():
    matrix = np.zeros((10, 20))
    rows, cols = matrix.shape
    assert rows == 10
    assert cols == 20


def test_counting():
    data = np.array([1, 2, 3, 4, 5])
    count = len(data)
    assert count == 5


def test_indexing():
    arr = np.arange(100)
    idx = np.argmax(arr)
    assert idx == 99


def validate_flags():
    flags = np.array([True, False, True])
    assert flags[0]
    return flags


result = validate_flags()
iterations = 100
assert iterations == 100
