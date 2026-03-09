import torch
import torch.nn as nn


class DynamicModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.fc_alt = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))

        if x.sum() > 0:
            return self.fc2(x)
        else:
            return self.fc_alt(x)


class StaticModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def optimize_dynamic_model(model, example_input):
    model.eval()
    scripted_model = torch.jit.script(model)
    return scripted_model


def optimize_static_model(model, example_input):
    model.eval()
    traced_model = torch.jit.trace(model, example_input)
    return traced_model


def script_dynamic_model(input_dim, hidden_dim, output_dim):
    model = DynamicModel(input_dim, hidden_dim, output_dim)
    model.eval()

    scripted = torch.jit.script(model)
    return scripted


def export_conditional_model(model, output_path):
    model.eval()

    scripted_model = torch.jit.script(model)
    scripted_model.save(output_path)
    return scripted_model


def trace_static_model():
    model = StaticModel(64, 128, 10)
    model.eval()

    example_input = torch.randn(1, 64)
    traced = torch.jit.trace(model, example_input)
    return traced
