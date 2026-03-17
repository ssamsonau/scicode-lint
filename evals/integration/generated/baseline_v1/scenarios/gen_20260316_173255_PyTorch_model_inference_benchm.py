import time

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class FeatureExtractor(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(*[ResBlock(64) for _ in range(4)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.stem(x)))
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def benchmark_model(
    model: nn.Module, device: torch.device, batch_size: int = 32, num_iters: int = 100
) -> dict:
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    latencies = []

    for _ in range(num_iters):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    return {
        "mean_ms": sum(latencies) / len(latencies),
        "min_ms": min(latencies),
        "throughput_fps": batch_size / (sum(latencies) / len(latencies) / 1000),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor(num_classes=1000).to(device)

    results = benchmark_model(model, device, batch_size=32, num_iters=100)
    print(f"Mean latency: {results['mean_ms']:.2f} ms")
    print(f"Min latency:  {results['min_ms']:.2f} ms")
    print(f"Throughput:   {results['throughput_fps']:.1f} fps")


if __name__ == "__main__":
    main()
