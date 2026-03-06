import torch
import torch.nn as nn


class NormalizationLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean_val = x.mean()
        std_val = x.std() + 1e-8
        x -= mean_val  # In-place subtraction
        x /= std_val  # In-place division
        return x * self.scale


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = NormalizationLayer()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        x = self.norm(x)
        return self.fc(x)


def compute_loss(model, inputs, labels):
    outputs = model(inputs)
    loss = nn.functional.cross_entropy(outputs, labels)
    return loss
