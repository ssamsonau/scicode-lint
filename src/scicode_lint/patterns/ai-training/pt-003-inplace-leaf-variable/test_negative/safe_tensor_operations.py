import torch
import torch.nn as nn


class SpectralNormFC(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        u = torch.randn(self.weight.shape[0], device=self.weight.device)
        v = torch.randn(self.weight.shape[1], device=self.weight.device)
        for _ in range(3):
            v = torch.mv(self.weight.t(), u)
            v = v / (v.norm() + 1e-8)
            u = torch.mv(self.weight, v)
            u = u / (u.norm() + 1e-8)
        sigma = u.dot(torch.mv(self.weight, v))
        normed_weight = self.weight / sigma
        return torch.nn.functional.linear(x, normed_weight, self.bias)


def clip_and_scale(tensor, max_norm=1.0):
    norm = tensor.norm()
    scale = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
    return tensor * scale
