import numpy as np
import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_model(checkpoint_path: str, input_dim: int) -> BinaryClassifier:
    model = BinaryClassifier(input_dim)
    state = torch.load(checkpoint_path)
    model.load_state_dict(state["model_state_dict"])
    return model


def run_inference(
    model: BinaryClassifier,
    features: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    x = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
    predictions = (logits > threshold).numpy().astype(int)
    return predictions


if __name__ == "__main__":
    checkpoint = "model_checkpoint.pt"
    input_dim = 32
    model = load_model(checkpoint, input_dim)
    sample_features = np.random.randn(16, input_dim).astype(np.float32)
    preds = run_inference(model, sample_features)
    print(f"Predictions: {preds}")
    print(f"Positive rate: {preds.mean():.2f}")
