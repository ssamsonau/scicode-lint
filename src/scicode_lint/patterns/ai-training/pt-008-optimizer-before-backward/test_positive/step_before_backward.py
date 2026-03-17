import torch.nn as nn
import torch.optim as optim


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


def train_feedforward(model, train_loader):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for data, labels in train_loader:
        outputs = model(data)
        loss = criterion(outputs, labels)

        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        optimizer.step()

        loss.backward()

    return model
