import torch
import torch.nn as nn


class CustomNormalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean()
        std = x.std()
        normalized = (x - mean) / (std + 1e-8)
        return self.gamma * normalized + self.beta


class DeepNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = CustomNormalization()
        self.layers = nn.Sequential(
            nn.Linear(200, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.layers(x)


def forward_pass(model, inputs):
    outputs = model(inputs)
    return outputs
