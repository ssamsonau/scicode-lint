import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.encoder(x)


def export_to_torchscript(model, output_path):
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(output_path)
    return scripted


def trace_and_save(model, example_input, output_path):
    model.eval()
    traced = torch.jit.trace(model, example_input)
    traced.save(output_path)
    return traced


class ProductionModel:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.model.eval()

    def predict(self, inputs):
        with torch.inference_mode():
            return self.model(inputs)
