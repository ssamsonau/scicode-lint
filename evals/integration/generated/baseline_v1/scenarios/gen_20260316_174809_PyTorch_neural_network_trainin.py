import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int = 10,
) -> list:
    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        total_loss = torch.tensor(0.0)
        for xb, yb in loader:
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss
        avg = total_loss / len(loader)
        epoch_losses.append(avg)
        print(f"Epoch {epoch + 1}/{epochs} loss={avg:.4f}")
    return epoch_losses


def main() -> None:
    X = torch.randn(1000, 16)
    y = torch.randint(0, 4, (1000,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MLP(16, 64, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses = train(model, loader, optimizer, criterion, epochs=5)
    print("Final loss:", losses[-1])


if __name__ == "__main__":
    main()
