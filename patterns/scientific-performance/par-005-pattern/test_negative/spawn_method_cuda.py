import multiprocessing as mp

import torch


def worker_function(rank, data_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.randn(data_size, data_size).to(device)
    result = tensor.sum()
    print(f"Worker {rank}: {result.item()}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    processes = []
    for i in range(4):
        p = mp.Process(target=worker_function, args=(i, 1000))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
