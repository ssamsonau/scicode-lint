import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.block1 = ResidualBlock(64)
        self.block2 = ResidualBlock(64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def create_model(num_classes, device="cpu"):
    model = ImageClassifier(num_classes)
    model = model.to(device)
    model.eval()
    return model


def initialize_pretrained_weights(model, pretrained_config):
    for name, param in model.named_parameters():
        if "conv" in name and "weight" in name:
            nn.init.kaiming_normal_(param)
        elif "bn" in name and "weight" in name:
            nn.init.ones_(param)
        elif "bn" in name and "bias" in name:
            nn.init.zeros_(param)
    return model
