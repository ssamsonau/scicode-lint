import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def save_scripted_model(model, path):
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(path)


def load_scripted_model(path, device="cpu"):
    model = torch.jit.load(path, map_location=device)
    return model


class InferenceEngine:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

    def predict(self, inputs):
        inputs = inputs.to(self.device)
        return self.model(inputs)


def load_traced_model(path):
    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    return model
