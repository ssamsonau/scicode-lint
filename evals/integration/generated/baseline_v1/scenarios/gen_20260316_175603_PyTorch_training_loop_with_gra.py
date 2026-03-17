import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb)
            total_loss += criterion(preds, yb).item()
    return total_loss / len(loader)


def train(model, train_loader, val_loader, epochs=5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        val_loss = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}: train={float(epoch_loss):.4f}, val={val_loss:.4f}")


if __name__ == "__main__":
    X = torch.randn(200, 16)
    y = torch.randn(200, 1)
    dataset = TensorDataset(X, y)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [160, 40])
    train_loader = DataLoader(train_ds, batch_size=32)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = MLP(16, 64, 1)
    train(model, train_loader, val_loader)
