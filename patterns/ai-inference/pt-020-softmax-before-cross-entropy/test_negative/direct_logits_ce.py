import torch.nn as nn
import torch.nn.functional as F


class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

    def compute_loss(self, images, labels):
        logits = self.forward(images)
        return self.criterion(logits, labels)


def train_classifier(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0

    for images, labels in dataloader:
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
