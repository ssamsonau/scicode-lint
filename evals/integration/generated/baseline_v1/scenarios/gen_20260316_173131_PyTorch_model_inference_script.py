from pathlib import Path

import torch
import torch.nn as nn


class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def load_model(checkpoint_path: str) -> ImageClassifier:
    model = ImageClassifier(num_classes=10)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    return model


def run_inference(model: ImageClassifier, data_path: str) -> dict:
    samples = torch.randn(64, 784)

    with torch.no_grad():
        logits = model(samples)

    predictions = (logits > 0.5).int()
    confidence = logits.softmax(dim=-1).max(dim=-1).values

    return {
        "predictions": predictions.tolist(),
        "confidence": confidence.tolist(),
        "num_samples": len(samples),
    }


def main() -> None:
    checkpoint = "checkpoints/classifier_epoch50.pt"
    if not Path(checkpoint).exists():
        print(f"Checkpoint not found: {checkpoint}")
        return

    model = load_model(checkpoint)
    results = run_inference(model, "data/test_set.npy")
    print(f"Processed {results['num_samples']} samples")
    print(f"Mean confidence: {sum(results['confidence']) / len(results['confidence']):.3f}")


if __name__ == "__main__":
    main()
