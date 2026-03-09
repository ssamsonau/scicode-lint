import numpy as np


def check_convergence(loss, threshold):
    return np.isclose(loss, threshold, rtol=1e-5)


def validate_results(computed, expected):
    return np.allclose(computed, expected, atol=1e-6)


class FloatValidator:
    def __init__(self, tolerance=1e-5):
        self.tolerance = tolerance

    def compare(self, a, b):
        return np.abs(a - b) < self.tolerance

    def verify_sum(self, data, expected):
        result = np.sum(data.astype(np.float32))
        return np.isclose(result, expected, rtol=self.tolerance)


def regression_test(model_output, baseline):
    diff = np.abs(model_output - baseline)
    max_diff = np.max(diff)
    return max_diff < 1e-4
