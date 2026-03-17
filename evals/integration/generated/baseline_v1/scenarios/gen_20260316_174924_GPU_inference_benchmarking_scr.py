import argparse
import time

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class ImageClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.blocks = nn.Sequential(*[ResBlock(64) for _ in range(4)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def benchmark_inference(batch_size=32, num_iters=100, device="cuda"):
    model = ImageClassifier().to(device)
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)

    latencies = []
    for i in range(num_iters):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    avg_latency = sum(latencies) / len(latencies)
    throughput = batch_size / (avg_latency / 1000)
    print(f"Avg latency: {avg_latency:.2f} ms")
    print(f"Throughput: {throughput:.1f} samples/sec")
    return avg_latency, throughput


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    benchmark_inference(args.batch_size, args.iters)
