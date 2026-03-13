import numpy as np


def test_sine_values():
    x = np.pi / 6
    result = np.sin(x)
    assert result == 0.5


def test_matrix_inverse():
    A = np.array([[1, 2], [3, 4]], dtype=float)
    A_inv = np.linalg.inv(A)
    product = A @ A_inv
    assert (product == np.eye(2)).all()


def test_normalization():
    vec = np.array([3.0, 4.0])
    normalized = vec / np.linalg.norm(vec)
    assert np.linalg.norm(normalized) == 1.0
