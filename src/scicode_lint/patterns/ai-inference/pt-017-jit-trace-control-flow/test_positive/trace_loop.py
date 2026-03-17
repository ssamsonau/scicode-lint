import torch
import torch.nn as nn


class RecurrentProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = nn.Linear(10, 10)
        self.threshold = 0.1

    def forward(self, x):
        num_iterations = int(x.abs().sum().item()) % 5 + 1
        for _ in range(num_iterations):
            x = self.transform(x)
        return x


def trace_loop_model():
    model = RecurrentProcessor()
    example = torch.randn(1, 10)
    return torch.jit.trace(model, example)
