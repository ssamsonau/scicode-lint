import time

import torch


def profile_layers(model, x):
    timings = {}
    x = x.cuda()

    start = time.time()
    x = model.conv1(x)
    timings["conv1"] = time.time() - start

    start = time.time()
    x = model.conv2(x)
    timings["conv2"] = time.time() - start

    return timings


def measure_forward_backward(model, x, y):
    x, y = x.cuda(), y.cuda()

    start = time.perf_counter()
    out = model(x)
    forward_time = time.perf_counter() - start

    start = time.perf_counter()
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    backward_time = time.perf_counter() - start

    return forward_time, backward_time
