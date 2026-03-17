import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def load_model(checkpoint_path: str) -> ImageClassifier:
    model = ImageClassifier(num_classes=10)
    state = torch.load(checkpoint_path)
    model.load_state_dict(state["model_state_dict"])
    return model


def run_inference(model: ImageClassifier, features: torch.Tensor, threshold: float = 0.5) -> dict:
    with torch.no_grad():
        logits = model(features)
        predictions = (logits > threshold).int()
        probs = torch.softmax(logits, dim=-1)
        top_class = torch.argmax(probs, dim=-1)
    return {"predictions": predictions, "top_class": top_class, "logits": logits}


def main() -> None:
    checkpoint = "checkpoints/best_model.pt"
    model = load_model(checkpoint)

    dummy_input = torch.randn(8, 512)
    results = run_inference(model, dummy_input)
    print(f"Top classes: {results['top_class']}")
    print(f"Predictions shape: {results['predictions'].shape}")


if __name__ == "__main__":
    main()
