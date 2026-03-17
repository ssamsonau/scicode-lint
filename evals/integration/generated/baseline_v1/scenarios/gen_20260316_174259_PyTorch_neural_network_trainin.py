import torch
import torch.nn as nn
import torch.optim as optim
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
            nn.ReLU(),
            ResidualBlock(128),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def train(epochs: int = 5) -> None:
    X = torch.randn(512, 32)
    y = torch.randint(0, 4, (512,))
    val_X = torch.randn(128, 32)
    val_y = torch.randint(0, 4, (128,))

    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    model = Classifier(32, 4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss

        model.eval()
        with torch.no_grad():
            val_logits = model(val_X)
            val_loss = criterion(val_logits, val_y)
            acc = (val_logits.argmax(1) == val_y).float().mean()

        print(
            f"Epoch {epoch + 1}: train_loss={float(epoch_loss):.4f} val_loss={val_loss:.4f} acc={acc:.4f}"
        )

        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    train()
