import torch
import torch.multiprocessing as mp


def worker(data):
    device = torch.device("cuda")
    return data.to(device).sum().item()


def main():
    mp.set_start_method("spawn", force=True)

    data = [torch.randn(100, 100) for _ in range(4)]

    with mp.Pool(4) as pool:
        results = pool.map(worker, data)
    return results


def init_and_parallelize():
    mp.set_start_method("spawn", force=True)
    data = [torch.randn(32, 10) for _ in range(8)]

    with mp.Pool(4) as pool:
        results = pool.map(worker, data)
    return results
