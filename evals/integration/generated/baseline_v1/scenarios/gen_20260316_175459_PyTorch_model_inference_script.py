from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ResidualClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = torch.relu(self.bn2(self.fc2(h)))
        return self.head(h)


def load_model(checkpoint_path: str) -> ResidualClassifier:
    model = ResidualClassifier(input_dim=128, hidden_dim=256, num_classes=1)
    state = torch.load(checkpoint_path)
    model.load_state_dict(state["model_state_dict"])
    return model


def run_inference(model: ResidualClassifier, features: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(features).float()
    with torch.no_grad():
        logits = model(x)
    predictions = (logits > 0.5).squeeze(1).numpy().astype(int)
    return predictions


def main() -> None:
    checkpoint_path = "checkpoints/classifier_epoch40.pt"
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model(checkpoint_path)

    rng = np.random.default_rng(42)
    test_features = rng.standard_normal((64, 128)).astype(np.float32)

    preds = run_inference(model, test_features)
    positive_rate = preds.mean()
    print(f"Positive prediction rate: {positive_rate:.3f}")
    print(f"Predictions: {preds[:10]}")


if __name__ == "__main__":
    main()
