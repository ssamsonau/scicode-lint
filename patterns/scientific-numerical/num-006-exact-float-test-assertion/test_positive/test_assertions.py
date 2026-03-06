import numpy as np


def test_optimization_result():
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    total = np.sum(weights)
    assert total == 1.0


def test_eigenvalues():
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    eigenvals = np.linalg.eigvalsh(matrix)
    expected = np.array([2.38196601, 4.61803399])
    assert (eigenvals == expected).all()


def test_gradient_descent():
    learning_rate = 0.01
    gradient = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.5, 0.3, 0.2])
    new_weights = weights - learning_rate * gradient
    expected = np.array([0.49, 0.28, 0.17])
    assert (new_weights == expected).all()


def validate_statistics(data):
    variance = np.var(data)
    assert variance != 0.0
    return variance


stats = validate_statistics(np.array([1.0, 2.0, 3.0]))
