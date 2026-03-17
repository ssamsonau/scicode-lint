import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)


def create_dataset(n_samples=1000):
    X = np.random.randn(n_samples, 10).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    return TensorDataset(torch.tensor(X), torch.tensor(y))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x).squeeze()


def train(model, loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


dataset = create_dataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()
train(model, loader, optimizer, criterion)
