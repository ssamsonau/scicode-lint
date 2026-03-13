import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)


class InferenceService:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device)
        self.model = ImageClassifier(num_classes=1000)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        logits = self.model(image_tensor)
        return logits.argmax(dim=1)

    def batch_predict(self, images):
        results = []
        with torch.inference_mode():
            for img in images:
                img = img.to(self.device)
                pred = self.model(img)
                results.append(pred.argmax(dim=1))
        return torch.cat(results)
