import time

import torch


def benchmark_with_warmup(model, inputs, warmup_iters=10, measure_iters=100):
    model.eval()
    model.cuda()
    inputs = inputs.cuda()

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(inputs)

    torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(measure_iters):
            _ = model(inputs)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / measure_iters


def profile_with_discard(model, data_loader, discard_first=5):
    model.eval()
    timings = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(batch.cuda())
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            if i >= discard_first:
                timings.append(elapsed)

    return sum(timings) / len(timings) if timings else 0
