"""
Machine learning pipeline for training and preprocessing.

This module implements data preprocessing and model training
with multiple function variations for different use cases.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class SimpleModel(nn.Module):
    """Basic neural network."""

    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def preprocess_features_v1(X_train, X_test):
    """Preprocess features using StandardScaler."""
    scaler = StandardScaler()
    all_data = np.vstack([X_train, X_test])
    scaler.fit(all_data)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def preprocess_features_v2(X_train, X_test):
    """Preprocess features using MinMaxScaler."""
    scaler = MinMaxScaler()
    combined = np.concatenate([X_train, X_test], axis=0)
    scaler.fit(combined)

    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm


def train_epoch_v1(model, data_loader, criterion, optimizer):
    """Train model for one epoch."""
    total_loss = 0.0

    for batch_data, batch_labels in data_loader:
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def train_epoch_v2(model, data_loader, criterion, optimizer, device):
    """Train model for one epoch with device support."""
    running_loss = 0.0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(data_loader)


def train_epoch_v3(model, train_data, train_labels, criterion, optimizer):
    """Train model on full batch."""
    model.train()

    outputs = model(train_data)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def generate_split_v1():
    """Generate synthetic data and split."""
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def generate_split_v2():
    """Generate classification data and split."""
    features = np.random.rand(500, 15)
    labels = np.random.randint(0, 2, (500, 1))
    train_X, test_X, train_y, test_y = train_test_split(features, labels, test_size=0.25)
    return train_X, test_X, train_y, test_y


def main():
    """Run the complete training pipeline."""
    print("Generating data...")
    X_train_1, X_test_1, y_train_1, y_test_1 = generate_split_v1()
    X_train_2, X_test_2, y_train_2, y_test_2 = generate_split_v2()

    print("Preprocessing features...")
    X_train_1, X_test_1 = preprocess_features_v1(X_train_1, X_test_1)
    X_train_2, X_test_2 = preprocess_features_v2(X_train_2, X_test_2)

    print("Training models...")
    model = SimpleModel(input_dim=X_train_1.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.FloatTensor(X_train_1), torch.FloatTensor(y_train_1))
    loader = DataLoader(dataset, batch_size=32)

    loss1 = train_epoch_v1(model, loader, criterion, optimizer)
    loss2 = train_epoch_v2(model, loader, criterion, optimizer, "cpu")
    loss3 = train_epoch_v3(
        model, torch.FloatTensor(X_train_1), torch.FloatTensor(y_train_1), criterion, optimizer
    )

    print(f"Training complete: {loss1:.4f}, {loss2:.4f}, {loss3:.4f}")


if __name__ == "__main__":
    main()
