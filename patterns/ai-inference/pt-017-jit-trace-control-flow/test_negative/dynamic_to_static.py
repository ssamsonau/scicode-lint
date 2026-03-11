import torch
import torch.nn as nn


class ConditionalModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        if out.mean() > 0:
            return torch.relu(self.fc2(out))
        return out


def compile_with_script(model):
    return torch.jit.script(model)


def export_model_correctly(model, has_control_flow=True):
    if has_control_flow:
        return torch.jit.script(model)
    return torch.jit.trace(model, torch.randn(1, 64))


class RouterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Linear(64, 32)
        self.branch_b = nn.Linear(64, 32)

    def forward(self, x, use_branch_a: bool):
        if use_branch_a:
            return self.branch_a(x)
        return self.branch_b(x)


# Use script for conditional model
scripted_router = torch.jit.script(RouterModel())
