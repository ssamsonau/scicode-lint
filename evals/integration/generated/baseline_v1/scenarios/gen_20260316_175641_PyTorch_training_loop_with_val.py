import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train(model, train_loader, val_loader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_correct += (preds.argmax(1) == yb).sum().item()
        val_acc = val_correct / len(val_loader.dataset)

        for xb, yb in train_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss

        print(f"Epoch {epoch + 1}: loss={float(total_loss):.4f}, val_acc={val_acc:.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(1000, 16)
    y = torch.randint(0, 4, (1000,))
    ds = TensorDataset(X, y)
    train_ds, val_ds = torch.utils.data.random_split(ds, [800, 200])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    model = MLP(16, 64, 4)
    train(model, train_loader, val_loader)
