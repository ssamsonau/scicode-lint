import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def load_model(checkpoint_path):
    model = ImageClassifier(num_classes=10)
    state = torch.load(checkpoint_path)
    model.load_state_dict(state["model_state_dict"])
    return model


def predict(model, image_path, threshold=0.5):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
    probs = logits.squeeze()
    predicted = (probs > threshold).int()
    return predicted, probs


if __name__ == "__main__":
    model = load_model("classifier.pth")
    images = [f for f in os.listdir("test_images") if f.endswith(".jpg")]
    for img_file in images:
        preds, scores = predict(model, os.path.join("test_images", img_file))
        print(f"{img_file}: {preds.tolist()}")
