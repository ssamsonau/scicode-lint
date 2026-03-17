import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            ResidualBlock(128),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))


def train(model, train_loader, val_loader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                correct += (logits.argmax(1) == y).sum().item()

        val_acc = correct / len(val_loader.dataset)
        print(
            f"Epoch {epoch + 1}: train_loss={float(train_loss):.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    X = torch.randn(1000, 32)
    y = torch.randint(0, 5, (1000,))
    ds = TensorDataset(X, y)
    train_ds, val_ds = torch.utils.data.random_split(ds, [800, 200])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    model = Classifier(32, 5)
    train(model, train_loader, val_loader)
