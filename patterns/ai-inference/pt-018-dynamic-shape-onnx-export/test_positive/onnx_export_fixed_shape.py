import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def export_model(model, output_path):
    model.eval()

    dummy_input = torch.randn(1, 128)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
    print(f"Model exported to {output_path}")


def deploy_model():
    model = SimpleClassifier(128, 64, 10)
    model.load_state_dict(torch.load("model.pth"))
    export_model(model, "model.onnx")
