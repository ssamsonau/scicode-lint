import time

import torch
import torch.nn as nn


class ResidualClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.layers(x))


def benchmark_model(model: nn.Module, dummy_input: torch.Tensor, n_runs: int = 200) -> dict:
    for _ in range(20):
        with torch.no_grad():
            model(dummy_input)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            out = model(dummy_input)
    end = time.perf_counter()

    avg_latency_ms = (end - start) / n_runs * 1000
    return {"avg_latency_ms": avg_latency_ms, "logits": out}


def classify_outputs(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (logits > threshold).long()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualClassifier(input_dim=1024, hidden_dim=512, num_classes=1).to(device)

    dummy_input = torch.randn(32, 1024, device=device)
    results = benchmark_model(model, dummy_input)

    print(f"Avg latency: {results['avg_latency_ms']:.2f} ms")

    logits = results["logits"]
    predictions = classify_outputs(logits)
    print(f"Positive predictions: {predictions.sum().item()} / {predictions.numel()}")


if __name__ == "__main__":
    main()
