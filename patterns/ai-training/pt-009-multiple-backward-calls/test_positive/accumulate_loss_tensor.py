import torch
import torch.nn as nn
import torch.optim as optim


class ImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 192, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
        )
        self.classifier = nn.Linear(192 * 6 * 6, 1000)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_with_tensor_accumulation(model, data_loader, epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = torch.tensor(0.0)

        for inputs, labels in data_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # BUG: Adding loss tensor directly - keeps computation graph
            epoch_loss = epoch_loss + loss

        _avg_loss = epoch_loss / len(data_loader)

    return model
