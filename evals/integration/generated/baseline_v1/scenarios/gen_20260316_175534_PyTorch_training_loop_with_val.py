import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class Classifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_classes=10):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.res = ResidualBlock(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.head(self.res(self.proj(x)))


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            running_loss += loss
            correct += (logits.argmax(1) == y).sum().item()
    return running_loss / len(loader), correct / len(loader.dataset)


def train(epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.randn(2000, 64)
    y = torch.randint(0, 10, (2000,))
    split = 1600
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds = TensorDataset(X[split:], y[split:])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = Classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()

    for epoch in range(epochs):
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss / len(train_loader):.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}"
        )


if __name__ == "__main__":
    train()
