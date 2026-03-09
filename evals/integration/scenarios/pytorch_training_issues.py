"""
Image classification model training and evaluation.

This module implements a CNN-based image classifier with
training, validation, and inference capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configuration (avoids rep-003 hardcoded hyperparameters)
CONFIG = {
    "num_classes": 10,
    "learning_rate": 0.001,
    "batch_size": 16,
    "num_epochs": 10,
    "dropout_rate": 0.5,
}

# Reproducibility setup (avoids rep-002 and rep-004)
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImageClassifier(nn.Module):
    """CNN for image classification."""

    def __init__(self, num_classes: int, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
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
    model.eval()
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


def load_pretrained(path):
    """Load a pretrained model checkpoint."""
    checkpoint = torch.load(path)
    model = ImageClassifier(
        num_classes=checkpoint["num_classes"],
        dropout_rate=checkpoint["dropout_rate"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


def benchmark_inference(model, input_tensor, num_runs=100):
    """Measure inference latency."""
    import time

    model.eval()
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(input_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def main():
    """Run training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageClassifier(
        num_classes=CONFIG["num_classes"],
        dropout_rate=CONFIG["dropout_rate"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    train_data = torch.randn(100, 3, 32, 32)
    train_labels = torch.randint(0, CONFIG["num_classes"], (100,))
    val_data = torch.randn(20, 3, 32, 32)
    val_labels = torch.randint(0, CONFIG["num_classes"], (20,))

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    for epoch in range(CONFIG["num_epochs"]):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    test_images = torch.randn(5, 3, 32, 32)
    predictions = predict_batch(model, test_images)
    print(f"\nPredictions: {predictions}")


if __name__ == "__main__":
    main()
