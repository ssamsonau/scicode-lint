import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


def train(model, loader, optimizer, criterion, accumulation_steps=4, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for step, (x, y) in enumerate(loader):
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")


def main():
    torch.manual_seed(0)
    X = torch.randn(800, 512)
    y = torch.randint(0, 5, (800,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SimpleClassifier(input_dim=512, num_classes=5)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train(model, loader, optimizer, criterion, accumulation_steps=4, epochs=3)


if __name__ == "__main__":
    main()
