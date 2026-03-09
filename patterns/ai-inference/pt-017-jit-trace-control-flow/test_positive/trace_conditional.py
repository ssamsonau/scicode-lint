import torch
import torch.nn as nn


class ConditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        if x.sum() > 0:
            return self.fc(x)
        return x


def export_model(model, example_input):
    traced = torch.jit.trace(model, example_input)
    return traced
