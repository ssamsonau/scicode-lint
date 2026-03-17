import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.bn(self.fc1(x)))))


def train_model(model, train_loader, val_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X, y in train_loader:
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                preds = model(X).argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        acc = val_correct / val_total
        print(f"Epoch {epoch + 1}: loss={epoch_loss:.4f} acc={acc:.3f}")


X = torch.randn(800, 128)
y = torch.randint(0, 10, (800,))
X_val = torch.randn(200, 128)
y_val = torch.randint(0, 10, (200,))

train_loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

model = SimpleNet()
train_model(model, train_loader, val_loader)
