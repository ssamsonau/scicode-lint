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


def evaluate_on_test_set(model, test_images, test_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_images = test_images.to(device)
    test_labels = test_labels.to(device)

    with torch.no_grad():
        predictions = model(test_images)
        accuracy = (predictions.argmax(dim=1) == test_labels).float().mean()

    return accuracy.item()


def run_inference(model, full_dataset):
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    data_tensor = torch.tensor(full_dataset).float().to(device)

    with torch.no_grad():
        outputs = model(data_tensor)

    return outputs.cpu().numpy()
