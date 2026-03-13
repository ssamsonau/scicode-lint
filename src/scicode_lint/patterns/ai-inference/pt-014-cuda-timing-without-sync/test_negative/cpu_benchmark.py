import time

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)


def benchmark_cpu_model(model, input_tensor, num_iterations=100):
    model.cpu()
    model.eval()
    input_tensor = input_tensor.cpu()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(input_tensor)
    end = time.perf_counter()

    return (end - start) / num_iterations


def measure_preprocessing_time(data, num_runs=50):
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        normalized = (data - data.mean()) / data.std()
        _result = torch.fft.fft(normalized)
        end = time.perf_counter()
        times.append(end - start)
    return sum(times) / len(times)


def profile_data_loading(dataloader):
    start = time.perf_counter()
    for batch in dataloader:
        pass
    end = time.perf_counter()
    return end - start
