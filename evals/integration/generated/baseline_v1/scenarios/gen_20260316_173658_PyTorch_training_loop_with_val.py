import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Classifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualBlock(128),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def run_training(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss

        print(f"Epoch {epoch + 1}, train loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                out = model(batch_x)
                val_loss += criterion(out, batch_y).item()
                correct += (out.argmax(1) == batch_y).sum().item()

        print(
            f"  val loss: {val_loss / len(val_loader):.4f}, acc: {correct / len(val_loader.dataset):.4f}"
        )


if __name__ == "__main__":
    X = torch.randn(1000, 32)
    y = torch.randint(0, 5, (1000,))
    train_ds = TensorDataset(X[:800], y[:800])
    val_ds = TensorDataset(X[800:], y[800:])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    model = Classifier(32, 5)
    run_training(model, train_loader, val_loader, epochs=5)
