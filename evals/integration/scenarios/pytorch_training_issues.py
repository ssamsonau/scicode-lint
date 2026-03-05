"""
Image classification model training and evaluation.

This module implements a CNN-based image classifier with
training, validation, and inference capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ImageClassifier(nn.Module):
    """CNN for image classification."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch."""
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion):
    """Validate the model."""
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader)

    return avg_loss, accuracy


def predict_batch(model, images):
    """Make predictions on a batch of images."""
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

    return predictions


def main():
    """Run training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageClassifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_data = torch.randn(100, 3, 32, 32)
    train_labels = torch.randint(0, 10, (100,))
    val_data = torch.randn(20, 3, 32, 32)
    val_labels = torch.randint(0, 10, (20,))

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    test_images = torch.randn(5, 3, 32, 32)
    predictions = predict_batch(model, test_images)
    print(f"\nPredictions: {predictions}")


if __name__ == "__main__":
    main()
