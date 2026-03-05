import torch
import torch.nn as nn
import torch.optim as optim


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1024, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.dropout(self.layer1(x)))
        x = torch.relu(self.dropout(self.layer2(x)))
        x = torch.relu(self.dropout(self.layer3(x)))
        return self.layer4(x)


def train_model(model, loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0

        for inputs, targets in loader:
            optimizer.zero_grad()

            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        _epoch_loss = running_loss / len(loader)

    return model
