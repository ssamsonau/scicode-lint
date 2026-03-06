"""Pytest configuration for scicode-lint tests."""

import pytest


@pytest.fixture
def sample_code_with_leakage() -> str:
    """Sample code with data leakage for testing."""
    return """
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
"""


@pytest.fixture
def sample_code_correct() -> str:
    """Sample correct code for testing."""
    return """
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""
