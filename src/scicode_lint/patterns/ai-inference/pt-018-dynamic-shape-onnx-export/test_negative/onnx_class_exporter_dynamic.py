import torch
import torch.nn as nn


class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, waveform):
        x = torch.relu(self.conv1(waveform))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def export_audio_model(model, output_path):
    model.eval()
    sample_waveform = torch.randn(1, 1, 16000)

    torch.onnx.export(
        model,
        sample_waveform,
        output_path,
        input_names=["waveform"],
        output_names=["class_scores"],
        dynamic_axes={
            "waveform": {0: "batch", 2: "samples"},
            "class_scores": {0: "batch"},
        },
        opset_version=13,
    )


def create_streaming_model(model, output_path):
    model.eval()
    chunk = torch.randn(1, 1, 4000)

    torch.onnx.export(
        model,
        chunk,
        output_path,
        input_names=["audio_chunk"],
        output_names=["predictions"],
        dynamic_axes={
            "audio_chunk": {0: "batch", 2: "chunk_length"},
            "predictions": {0: "batch"},
        },
    )
