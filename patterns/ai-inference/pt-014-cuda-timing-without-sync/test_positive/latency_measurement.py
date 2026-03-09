import time

import torch


def profile_forward_pass(model, inputs):
    model.cuda()
    inputs = inputs.cuda()

    start = time.perf_counter()
    outputs = model(inputs)
    elapsed = time.perf_counter() - start

    return elapsed, outputs


def compare_batch_sizes(model, input_shape):
    times = {}
    for batch_size in [1, 8, 32, 128]:
        x = torch.randn(batch_size, *input_shape).cuda()
        start = time.time()
        _ = model(x)
        times[batch_size] = time.time() - start
    return times
