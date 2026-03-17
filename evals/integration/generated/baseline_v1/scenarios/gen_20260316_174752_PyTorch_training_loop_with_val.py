import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.features(x)


def train(num_epochs=5, batch_size=32):
    X = torch.randn(1000, 128)
    y = torch.randint(0, 10, (1000,))
    X_val = torch.randn(200, 128)
    y_val = torch.randint(0, 10, (200,))

    train_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    model = ConvNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                val_loss += criterion(out, yb).item()
        print(f"Epoch {epoch} val_loss={val_loss / len(val_loader):.4f}")

        epoch_loss = 0.0
        for xb, yb in train_loader:
            out = model(xb)
            loss = criterion(out, yb)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} train_loss={epoch_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    train()
