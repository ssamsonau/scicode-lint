import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(32 * 26 * 26, 10)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv1(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def evaluate_model(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    # BUG: Missing model.eval() - dropout and batch norm stay in training mode
    for images, labels in test_loader:
        # Disabling gradients but not setting eval mode
        outputs = model(images)
        outputs.detach_()

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy, total_loss
