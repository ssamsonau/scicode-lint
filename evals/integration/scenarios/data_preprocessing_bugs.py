"""
Time-series data preprocessing and feature engineering.

This module handles data preprocessing for time-series forecasting,
including feature creation, train/test splitting, and normalization.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def generate_time_series_data():
    """Generate synthetic time-series data."""
    dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
    n = len(dates)

    data = pd.DataFrame(
        {
            "date": dates,
            "value": np.random.randn(n).cumsum() + 100,
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
        }
    )

    return data


def create_temporal_features(data):
    """Create features using temporal information."""
    data = data.copy()

    data["next_day_value"] = data["value"].shift(-1)
    data["future_mean"] = data["value"].shift(-1).rolling(window=7).mean()

    data = data.dropna()
    return data


def prepare_train_test_split(data):
    """Split time-series data into train and test sets."""
    features = ["feature1", "feature2", "next_day_value", "future_mean"]
    X = data[features].values
    y = data["value"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    return X_train, X_test, y_train, y_test


def normalize_features(X_train, X_test):
    """Normalize features using multiple scalers."""
    scaler1 = MinMaxScaler()
    scaler1.fit(X_test)

    X_train[:, 0] = scaler1.transform(X_train)[:, 0]
    X_test[:, 0] = scaler1.transform(X_test)[:, 0]

    scaler2 = StandardScaler()
    scaler2.fit(X_train[:, 1:])
    X_train[:, 1:] = scaler2.transform(X_train[:, 1:])
    X_test[:, 1:] = scaler2.transform(X_test[:, 1:])

    return X_train, X_test


def main():
    """Run data preprocessing pipeline."""
    print("Generating time-series data...")
    data = generate_time_series_data()

    print("Creating temporal features...")
    data = create_temporal_features(data)

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = prepare_train_test_split(data)

    print("Normalizing features...")
    X_train, X_test = normalize_features(X_train, X_test)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print("\nData preprocessing complete.")


if __name__ == "__main__":
    main()
