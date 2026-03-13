import time

import torch


def measure_throughput(model, batch_size, num_batches=100):
    model.cuda()
    model.eval()

    inputs = torch.randn(batch_size, 3, 224, 224).cuda()

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_batches):
        with torch.no_grad():
            _ = model(inputs)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    return (num_batches * batch_size) / elapsed


def time_single_batch(model, x):
    x = x.cuda()
    start = time.perf_counter()
    model(x)
    torch.cuda.synchronize()
    return time.perf_counter() - start
