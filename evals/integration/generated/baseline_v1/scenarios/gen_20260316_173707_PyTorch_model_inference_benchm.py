import argparse
import time

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x)))) + x


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = ResBlock(64)
    model.load_state_dict(checkpoint["model_state"])
    return model


def benchmark_inference(model, device, num_iterations=100, batch_size=8):
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(batch_size, 64, 128, 128, device=device)

    latencies = []
    with torch.no_grad():
        start_total = time.perf_counter()
        for i in range(num_iterations):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
        end_total = time.perf_counter()

    mean_lat = sum(latencies) / len(latencies)
    throughput = (num_iterations * batch_size) / (end_total - start_total)
    print(f"Mean latency: {mean_lat:.2f} ms")
    print(f"Throughput: {throughput:.1f} samples/sec")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint)
    benchmark_inference(model, device, num_iterations=args.iters)


if __name__ == "__main__":
    main()
