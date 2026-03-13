import torch
import torch.nn as nn


class ConditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        if x.sum() > 0:
            return self.fc1(x)
        else:
            return self.fc2(x)


class VariableLengthModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x, num_layers):
        for _ in range(num_layers):
            x = self.fc(x)
        return x


def trace_model():
    model = ConditionalModel()
    example_input = torch.randn(1, 10)
    traced = torch.jit.trace(model, example_input)
    return traced


def trace_variable_model():
    model = VariableLengthModel()
    example_input = torch.randn(1, 10)
    traced = torch.jit.trace(model, (example_input, 3))
    return traced
