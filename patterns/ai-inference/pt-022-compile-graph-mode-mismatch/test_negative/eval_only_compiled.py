import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def create_inference_service(model_path, device="cuda"):
    model = ImageClassifier(10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    compiled = torch.compile(model)
    return compiled


def run_inference(compiled_model, images):
    with torch.inference_mode():
        return compiled_model(images)


class ProductionInference:
    def __init__(self, checkpoint_path):
        model = ImageClassifier(10)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        self.model = torch.compile(model)

    def __call__(self, inputs):
        with torch.inference_mode():
            return self.model(inputs)
