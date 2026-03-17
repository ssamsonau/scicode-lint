import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TumorClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def run_inference(
    model: TumorClassifier,
    features: torch.Tensor,
    threshold: float = 0.5,
    batch_size: int = 256,
) -> torch.Tensor:
    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=batch_size)
    predictions = []
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch)
            preds = (logits > threshold).long()
            predictions.append(preds)
    return torch.cat(predictions)


def main() -> None:
    model = TumorClassifier(input_dim=30)
    checkpoint = torch.load("tumor_classifier.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    features = torch.randn(1000, 30)
    labels = run_inference(model, features)

    positive_rate = labels.float().mean().item()
    print(f"Predicted positive rate: {positive_rate:.3f}")
    print(f"Positive samples: {labels.sum().item()} / {len(labels)}")


if __name__ == "__main__":
    main()
