import multiprocessing as mp

import torch


def worker(data):
    return data.sum().item()


def main():
    device = torch.device("cuda")
    tensor = torch.randn(100, 100).to(device)

    with mp.Pool(4) as pool:
        results = pool.map(worker, [tensor.cpu() for _ in range(4)])
    return results


def init_and_parallelize():
    model = torch.nn.Linear(10, 10).cuda()
    data = [torch.randn(32, 10) for _ in range(8)]

    with mp.Pool(4) as pool:
        results = pool.map(lambda x: model(x.cuda()).sum().item(), data)
    return results
