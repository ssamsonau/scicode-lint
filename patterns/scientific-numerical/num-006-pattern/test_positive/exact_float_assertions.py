import unittest

import numpy as np


class TestNumericalComputation(unittest.TestCase):
    def test_matrix_multiplication(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[2.0, 0.0], [1.0, 2.0]])
        result = A @ B
        expected = np.array([[4.0, 4.0], [10.0, 8.0]])
        assert (result == expected).all()

    def test_trigonometric(self):
        x = np.pi / 4
        result = np.sin(x) ** 2 + np.cos(x) ** 2
        assert result == 1.0

    def test_mean_calculation(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = np.mean(data)
        self.assertEqual(mean, 3.0)


def verify_computation():
    value = 0.1 + 0.2
    assert value == 0.3


result = np.sqrt(2.0) * np.sqrt(2.0)
assert result == 2.0
