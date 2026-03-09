import numpy as np
import pytest


def test_sine_values():
    x = np.pi / 6
    result = np.sin(x)
    assert np.isclose(result, 0.5, rtol=1e-10)


def test_matrix_inverse():
    A = np.array([[1, 2], [3, 4]], dtype=float)
    A_inv = np.linalg.inv(A)
    product = A @ A_inv
    np.testing.assert_allclose(product, np.eye(2), atol=1e-10)


def test_normalization():
    vec = np.array([3.0, 4.0])
    normalized = vec / np.linalg.norm(vec)
    assert pytest.approx(np.linalg.norm(normalized), rel=1e-10) == 1.0
