"""Example buggy code for testing scicode-lint."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data leakage bug: scaler is fit on full dataset before split
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

# BUG: Fitting scaler on full data including test set
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # This is wrong!

# Split happens AFTER scaling - test data leaked into training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Training model...")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
