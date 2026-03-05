import multiprocessing as mp

import torch


def worker_process(rank, tensor):
    result = tensor * rank
    print(f"Worker {rank}: {result.sum()}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.randn(1000, 1000).to(device)

    processes = []
    for i in range(4):
        p = mp.Process(target=worker_process, args=(i, data))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
