import numpy as np
from scipy import stats


def test_correlation_coefficient():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * x + 1.0
    r, _ = stats.pearsonr(x, y)
    assert r == 1.0


def test_standard_deviation():
    data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    std = np.std(data, ddof=1)
    assert std == 2.0


def test_probability_sum():
    probs = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
    total = np.sum(probs)
    assert total == 1.0
