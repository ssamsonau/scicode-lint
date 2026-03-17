import time

import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class BenchmarkModel(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(4)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def benchmark_inference(
    model: nn.Module,
    batch_size: int = 32,
    num_iterations: int = 100,
    device: str = "cuda",
) -> dict[str, float]:
    model = model.to(device)
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)

    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    latencies_arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies_arr)),
        "p50_ms": float(np.percentile(latencies_arr, 50)),
        "p95_ms": float(np.percentile(latencies_arr, 95)),
        "p99_ms": float(np.percentile(latencies_arr, 99)),
        "throughput_fps": float(batch_size * 1000 / np.mean(latencies_arr)),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BenchmarkModel(num_classes=1000)

    results = benchmark_inference(model, batch_size=32, num_iterations=100, device=device)

    print(f"Inference Benchmark Results (device={device}):")
    for metric, value in results.items():
        print(f"  {metric}: {value:.2f}")


if __name__ == "__main__":
    main()
