import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    X = torch.randn(512, 20)
    y = (X[:, 0] + X[:, 1] > 0).long()
    ds = TensorDataset(X, y)
    train_ds, val_ds = torch.utils.data.random_split(ds, [400, 112])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = MLP(20, 64, 2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        tr_loss = train(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}: train={tr_loss:.4f} val={val_loss:.4f}")


if __name__ == "__main__":
    main()
