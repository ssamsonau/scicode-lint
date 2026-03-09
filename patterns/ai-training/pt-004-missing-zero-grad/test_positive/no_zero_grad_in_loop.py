import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train(model, train_loader, epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        for batch_data, batch_labels in train_loader:
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()

    return model
