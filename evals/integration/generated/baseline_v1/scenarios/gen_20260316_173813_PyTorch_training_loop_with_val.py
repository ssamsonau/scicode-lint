import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            ResidualBlock(128),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


def train(model, loader, optimizer, criterion):
    total_loss = 0.0
    for xb, yb in loader:
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            val_loss += criterion(out, yb).item()
    return val_loss / len(loader)


def main():
    X = torch.randn(1000, 64)
    y = torch.randint(0, 10, (1000,))
    ds = TensorDataset(X, y)
    train_loader = DataLoader(ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(ds, batch_size=64)

    model = Classifier(64, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        tr_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}: train={tr_loss:.4f} val={val_loss:.4f}")


if __name__ == "__main__":
    main()
