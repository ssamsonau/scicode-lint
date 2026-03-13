import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)


def export_to_onnx(model, path):
    dummy = torch.randn(1, 100)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["output"])
