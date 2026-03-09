import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


def export_to_onnx(model, path, input_shape):
    model.eval()
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )


def save_scripted_model(model, path):
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(path)


def save_traced_model(model, example_input, path):
    model.eval()
    traced = torch.jit.trace(model, example_input)
    traced.save(path)
