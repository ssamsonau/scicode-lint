import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


def save_pytorch_model(model, path):
    torch.save(model.state_dict(), path)


def save_checkpoint(model, optimizer, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_pytorch_model(model_class, path, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


class ModelCheckpointer:
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def save(self, model, name):
        path = f"{self.save_dir}/{name}.pt"
        torch.save(model.state_dict(), path)

    def load(self, model, name):
        path = f"{self.save_dir}/{name}.pt"
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
