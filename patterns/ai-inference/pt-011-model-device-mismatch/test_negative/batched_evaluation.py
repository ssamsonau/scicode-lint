import torch
import torch.nn as nn


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 7 * 7, num_classes)

    def forward(self, x):
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)


def evaluate_on_test_set(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            correct += (predictions.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy


def run_inference(model, data_loader):
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    all_outputs = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            outputs = model(batch)
            all_outputs.append(outputs.cpu())

    return torch.cat(all_outputs, dim=0).numpy()
