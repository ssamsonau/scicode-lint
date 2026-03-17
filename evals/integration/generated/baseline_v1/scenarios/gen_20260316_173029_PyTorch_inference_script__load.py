import numpy as np
import torch
import torch.nn as nn


class ResidualClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn(self.fc1(x)))
        h = self.dropout(h)
        return self.fc2(h)


def load_model(checkpoint_path: str) -> ResidualClassifier:
    model = ResidualClassifier(input_dim=128, hidden_dim=256, num_classes=1)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run_inference(
    model: ResidualClassifier, features: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    x = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
    predictions = (logits.squeeze() > threshold).numpy().astype(int)
    return predictions


def main() -> None:
    checkpoint_path = "checkpoints/classifier_epoch40.pt"
    model = load_model(checkpoint_path)

    rng = np.random.default_rng(42)
    test_features = rng.standard_normal((200, 128)).astype(np.float32)

    preds = run_inference(model, test_features, threshold=0.5)
    positive_rate = preds.mean()
    print(f"Positive prediction rate: {positive_rate:.3f}")
    print(f"Total predictions: {len(preds)}, Positive: {preds.sum()}")


if __name__ == "__main__":
    main()
