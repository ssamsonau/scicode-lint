import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ONNXExporter:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def export_for_deployment(self, output_path):
        sample_input = torch.randn(1, 3, 224, 224)

        torch.onnx.export(
            self.model,
            sample_input,
            output_path,
            input_names=["images"],
            output_names=["logits"],
        )
