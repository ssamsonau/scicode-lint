import time

import torch


def benchmark_cpu_model(model, inputs, num_iterations=100):
    model.cpu()
    model.eval()
    inputs = inputs.cpu()

    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            model(inputs)
            end = time.perf_counter()
            times.append(end - start)

    return sum(times) / len(times)


def compare_batch_sizes(model, batch_sizes, feature_dim=256):
    model.cpu()
    model.eval()
    results = {}

    for bs in batch_sizes:
        inputs = torch.randn(bs, feature_dim)
        avg_time = benchmark_cpu_model(model, inputs)
        results[bs] = avg_time

    return results


def measure_preprocessing_time(transform, data, num_runs=50):
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = transform(data)
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)
