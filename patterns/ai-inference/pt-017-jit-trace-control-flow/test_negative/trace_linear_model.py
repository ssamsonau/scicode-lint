import torch
import torch.nn as nn


class LinearRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class FixedMLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x):
        return self.layers(x)


def trace_simple_model(model, example_input):
    model.eval()
    traced = torch.jit.trace(model, example_input)
    return traced


def export_encoder(encoder, sample_input):
    return torch.jit.trace(encoder, sample_input)
