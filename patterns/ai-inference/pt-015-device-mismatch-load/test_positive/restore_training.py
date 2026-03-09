import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x.mean(dim=[2, 3])
        return self.fc(x)


def restore_from_checkpoint(path):
    checkpoint = torch.load(path)
    model = ResNet(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["epoch"]


def load_pretrained_weights(model, weights_path):
    weights = torch.load(weights_path)
    model.load_state_dict(weights, strict=False)
    return model
