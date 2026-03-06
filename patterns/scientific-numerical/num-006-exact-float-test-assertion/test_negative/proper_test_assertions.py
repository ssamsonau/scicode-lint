import unittest

import numpy as np


class TestNumericalComputation(unittest.TestCase):
    def test_matrix_multiplication(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[2.0, 0.0], [1.0, 2.0]])
        result = A @ B
        expected = np.array([[4.0, 4.0], [10.0, 8.0]])
        np.testing.assert_allclose(result, expected)

    def test_trigonometric(self):
        x = np.pi / 4
        result = np.sin(x) ** 2 + np.cos(x) ** 2
        np.testing.assert_allclose(result, 1.0, rtol=1e-10)

    def test_mean_calculation(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = np.mean(data)
        np.testing.assert_allclose(mean, 3.0)


def verify_computation():
    value = 0.1 + 0.2
    np.testing.assert_allclose(value, 0.3)


result = np.sqrt(2.0) * np.sqrt(2.0)
np.testing.assert_allclose(result, 2.0, atol=1e-9)
