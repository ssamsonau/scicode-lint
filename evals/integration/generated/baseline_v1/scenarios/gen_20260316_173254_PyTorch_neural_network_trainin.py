import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(model: nn.Module, loader: DataLoader, epochs: int = 5) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    epoch_losses: list[float] = []

    for epoch in range(epochs):
        total_loss = torch.tensor(0.0)
        for xb, yb in loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss
        avg = total_loss / len(loader)
        epoch_losses.append(avg.item())
        print(f"Epoch {epoch + 1}: loss={avg.item():.4f}")

    return epoch_losses


def main() -> None:
    torch.manual_seed(42)
    X = torch.randn(1000, 16)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    model = MLP(16, 64, 1)
    losses = train(model, loader)
    print(f"Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
