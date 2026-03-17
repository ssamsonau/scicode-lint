import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.features(x)


def train(model, train_loader, val_loader, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_out = model(val_x)
                val_correct += (val_out.argmax(1) == val_y).sum().item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, val_acc={val_correct}")

        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    X = torch.randn(500, 128)
    y = torch.randint(0, 10, (500,))
    ds = TensorDataset(X, y)
    train_ds, val_ds = torch.utils.data.random_split(ds, [400, 100])
    train_loader = DataLoader(train_ds, batch_size=32)
    val_loader = DataLoader(val_ds, batch_size=32)
    model = ConvNet()
    train(model, train_loader, val_loader)
