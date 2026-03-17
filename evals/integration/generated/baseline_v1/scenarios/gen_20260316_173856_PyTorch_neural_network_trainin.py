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


def train(epochs: int = 5) -> None:
    X = torch.randn(1000, 20)
    y = torch.randint(0, 4, (1000,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    model = MLP(20, 64, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0

        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()

        avg_loss = total_loss / len(loader)
        acc = correct / len(dataset)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, acc={acc:.4f}")


if __name__ == "__main__":
    train()
