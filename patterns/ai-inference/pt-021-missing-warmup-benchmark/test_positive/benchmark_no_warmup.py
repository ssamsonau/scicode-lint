import time

import torch


def benchmark_model(model, input_tensor, num_iterations=100):
    model.eval()
    model.cuda()
    input_tensor = input_tensor.cuda()

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(num_iterations):
        with torch.no_grad():
            model(input_tensor)

    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / num_iterations


def measure_latency(model, x):
    start = time.time()
    model(x)
    torch.cuda.synchronize()
    return time.time() - start
