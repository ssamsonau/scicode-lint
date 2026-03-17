import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class TumorClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model(checkpoint_path):
    model = TumorClassifier()
    state = torch.load(checkpoint_path)
    model.load_state_dict(state["model_state_dict"])
    return model


def preprocess(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def predict(model, image_path, threshold=0.5):
    tensor = preprocess(image_path)
    with torch.no_grad():
        logits = model(tensor)
    predicted = (logits > threshold).item()
    return "malignant" if predicted else "benign"


if __name__ == "__main__":
    checkpoint = sys.argv[1]
    image = sys.argv[2]
    model = load_model(checkpoint)
    label = predict(model, image)
    print(f"Prediction: {label}")
