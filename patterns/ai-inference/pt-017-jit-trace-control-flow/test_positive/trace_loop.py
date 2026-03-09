import torch
import torch.nn as nn


class ConditionalModel(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.linear = nn.Linear(10, 10)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        if x.mean() > self.threshold:
            x = self.activation(x)
        return x


def trace_conditional_model():
    model = ConditionalModel()
    example = torch.randn(1, 10)
    return torch.jit.trace(model, example)
