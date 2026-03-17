import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    torch.manual_seed(42)
    X = torch.randn(512, 20)
    y = (X[:, 0] + X[:, 1] > 0).long()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = BinaryClassifier(input_dim=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(15):
        loss = train(model, loader, optimizer, criterion)
        print(f"Epoch {epoch + 1}/15  loss={loss:.4f}")


if __name__ == "__main__":
    main()
