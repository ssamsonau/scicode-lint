"""Medium scientific script: ML training pipeline with data processing (~200 lines)."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class FeatureProcessor:
    """Handles feature engineering and preprocessing."""

    def __init__(self, numerical_cols, categorical_cols):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.scaler = StandardScaler()
        self.encoders = {}

    def fit_transform(self, X):
        X_processed = X.copy()

        for col in self.categorical_cols:
            encoder = LabelEncoder()
            X_processed[:, col] = encoder.fit_transform(X_processed[:, col])
            self.encoders[col] = encoder

        X_processed[:, self.numerical_cols] = self.scaler.fit_transform(
            X_processed[:, self.numerical_cols].astype(float)
        )
        return X_processed.astype(float)

    def transform(self, X):
        X_processed = X.copy()
        for col in self.categorical_cols:
            X_processed[:, col] = self.encoders[col].transform(X_processed[:, col])

        X_processed[:, self.numerical_cols] = self.scaler.transform(
            X_processed[:, self.numerical_cols].astype(float)
        )
        return X_processed.astype(float)


class SimpleClassifier(nn.Module):
    """Feed-forward neural network for classification."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.network(x)


def create_synthetic_dataset(n_samples=10000, n_features=20, n_classes=5):
    """Generate a synthetic classification dataset."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)

    weights = np.random.randn(n_features, n_classes)
    logits = X @ weights
    y = np.argmax(logits, axis=1)

    noise_idx = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    y[noise_idx] = np.random.randint(0, n_classes, size=len(noise_idx))

    return X, y


def prepare_data(X, y, test_size=0.2, val_size=0.1):
    """Split data into train/val/test sets."""
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=64):
    """Create PyTorch DataLoaders."""
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


def evaluate(model, val_loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    return total_loss / total, correct / total


def compute_class_weights(y):
    """Compute inverse frequency class weights."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = total / (len(classes) * counts)
    return torch.FloatTensor(weights)


def run_experiment(n_epochs=50, learning_rate=0.001, hidden_dim=128):
    """Run full training experiment."""
    X, y = create_synthetic_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(X, y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(np.unique(y))
    model = SimpleClassifier(X_train.shape[1], hidden_dim, n_classes).to(device)

    class_weights = compute_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    best_val_acc = 0.0
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{n_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

    model.load_state_dict(torch.load("best_model.pt"))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=64)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    return model, test_acc


if __name__ == "__main__":
    model, accuracy = run_experiment()
