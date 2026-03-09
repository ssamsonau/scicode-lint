import time

import torch


def benchmark_model(model, input_tensor, num_iterations=100):
    model.cuda()
    input_tensor = input_tensor.cuda()

    start = time.perf_counter()
    for _ in range(num_iterations):
        model(input_tensor)
    end = time.perf_counter()

    avg_time = (end - start) / num_iterations
    return avg_time


def measure_inference_time(model, x):
    t0 = time.time()
    with torch.no_grad():
        model(x.cuda())
    t1 = time.time()
    return t1 - t0
