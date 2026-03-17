import numpy as np
import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
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


def run_inference(model: BinaryClassifier, features: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(features).float()
    with torch.no_grad():
        logits = model(x)
    predictions = (logits > 0.5).int().numpy()
    return predictions


def main() -> None:
    checkpoint = "checkpoints/classifier_final.pt"
    data = np.load("data/test_features.npy")

    model = load_model(checkpoint, input_dim=data.shape[1])

    preds = run_inference(model, data)
    positive_rate = preds.mean()
    print(f"Positive rate: {positive_rate:.3f}")
    print(f"Predictions shape: {preds.shape}")


if __name__ == "__main__":
    main()
