import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)


def load_model(path):
    model = SimpleModel()
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = SimpleModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["epoch"]
