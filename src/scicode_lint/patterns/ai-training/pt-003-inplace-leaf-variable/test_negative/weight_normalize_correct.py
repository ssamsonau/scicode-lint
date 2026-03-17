import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize


class SpectralNormLinear(nn.Module):
    """Linear layer with spectral normalization."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        spec_norm = nn.utils.parametrizations.spectral_norm(self.linear)
        parametrize.register_parametrization(
            self.linear, "weight", spec_norm.parametrizations.weight[0]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FunctionalLayerNorm(nn.Module):
    """Layer normalization using functional API."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


def update_weights(model: nn.Module, updates: dict[str, torch.Tensor]):
    """Update weights without in-place operations on leaf tensors."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in updates:
                param.copy_(updates[name])


def compute_weight_norm_stats(model: nn.Module) -> dict[str, float]:
    """Compute weight statistics without modifying parameters."""
    stats = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            stats[f"{name}_norm"] = param.norm().item()
            stats[f"{name}_mean"] = param.mean().item()
    return stats
