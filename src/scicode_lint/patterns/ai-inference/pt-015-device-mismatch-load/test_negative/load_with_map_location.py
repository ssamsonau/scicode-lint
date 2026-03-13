import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)


def load_model_portable(path, device="cpu"):
    """Load model with map_location for portable deployment."""
    model = SimpleModel()
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)


def load_checkpoint_to_cpu(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SimpleModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["epoch"]


def load_to_available_device(path):
    """Load model to GPU if available, otherwise CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(path, map_location=device)
    return state_dict
