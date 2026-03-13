import torch
import torch.nn as nn


class LayerWithInplaceOp(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 5))

    def forward(self, x):
        self.weight += torch.zeros_like(self.weight)
        return torch.matmul(x, self.weight)


def train_with_inplace_param_modification(model, data, target):
    output = model(data)
    loss = ((output - target) ** 2).mean()
    loss.backward()
    return loss.item()
