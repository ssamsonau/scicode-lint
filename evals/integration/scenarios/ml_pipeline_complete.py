"""
Complete ML pipeline for price prediction.

This module implements an end-to-end machine learning pipeline
for predicting prices using neural networks.
"""

import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler


class PredictionModel(nn.Module):
    """Simple neural network for regression."""

    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_and_preprocess_data():
    """Load data and create features."""
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    y = np.random.randn(n_samples, 1)

    price_category = (y > y.mean()).astype(int)
    X = np.column_stack([X, price_category])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test


def normalize_data(X_train, X_test):
    """Normalize features using StandardScaler."""
    scaler = StandardScaler()

    all_data = np.vstack([X_train, X_test])
    scaler.fit(all_data)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def train_model(model, X_train, y_train, epochs=50):
    """Train the neural network."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train)

    for epoch in range(epochs):
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        y_tensor = torch.FloatTensor(y_test)
        predictions = model(X_tensor)
        mse = nn.MSELoss()(predictions, y_tensor)

    return mse.item()


def load_data_files(data_dir):
    """Load all CSV files from a directory."""
    all_data = []
    for filepath in glob.glob(f"{data_dir}/*.csv"):
        df = pd.read_csv(filepath)
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


def sample_training_data(df, frac=0.1):
    """Sample a fraction of the data for quick experiments."""
    return df.sample(frac=frac)


def cross_validate(X, y, n_splits=5):
    """Perform k-fold cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_train_cv, X_val = X[train_idx], X[val_idx]
        y_train_cv, y_val = y[train_idx], y[val_idx]
        model = PredictionModel(input_dim=X_train_cv.shape[1])
        model = train_model(model, X_train_cv, y_train_cv, epochs=10)
        mse = evaluate_model(model, X_val, y_val)
        scores.append(mse)
    return np.mean(scores)


def main():
    """Run the complete ML pipeline."""
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    print("Normalizing features...")
    X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)

    print("Training model...")
    model = PredictionModel(input_dim=X_train_scaled.shape[1])
    model = train_model(model, X_train_scaled, y_train)

    print("Evaluating model...")
    test_mse = evaluate_model(model, X_test_scaled, y_test)
    print(f"Test MSE: {test_mse:.4f}")


if __name__ == "__main__":
    main()
