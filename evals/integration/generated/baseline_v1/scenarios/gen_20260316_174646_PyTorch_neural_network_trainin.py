import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 10)
        )
        self.bn = nn.BatchNorm1d(64)

    def forward(self, x):
        h = self.features[0](x)
        h = self.bn(self.features[1](h))
        h = self.features[2](h)
        return self.features[3](h)


X = torch.randn(500, 128)
y = torch.randint(0, 10, (500,))
ds = TensorDataset(X, y)
train_loader = DataLoader(ds, batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X[:100], y[:100]), batch_size=32)

model = ConvNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss

    if epoch % 2 == 0:
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                val_loss += criterion(out, yb).item()
                correct += (out.argmax(1) == yb).sum().item()
        print(f"Epoch {epoch} | train_loss={epoch_loss:.4f} | val_acc={correct / 100:.3f}")

    for xb, yb in train_loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
