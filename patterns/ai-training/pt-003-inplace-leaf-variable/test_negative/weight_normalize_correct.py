import torch
import torch.nn as nn


class WeightNormalizedLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        normalized_weight = self.weight / self.weight.norm(dim=1, keepdim=True)
        return x @ normalized_weight.T + self.bias


def apply_weight_decay(model, decay_rate=0.01):
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(1 - decay_rate)
