import multiprocessing as mp

import torch
from torch.utils.data import DataLoader, TensorDataset


def worker_fn(data_queue, device):
    model = torch.nn.Linear(10, 5).to(device)
    while True:
        batch = data_queue.get()
        if batch is None:
            break
        batch = batch.to(device)
        _ = model(batch)


def main():
    device = "cuda"

    dataset = TensorDataset(torch.randn(1000, 10))
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

    data_queue = mp.Queue()
    workers = []

    for _ in range(4):
        p = mp.Process(target=worker_fn, args=(data_queue, device))
        p.start()
        workers.append(p)

    for batch in dataloader:
        data_queue.put(batch[0])

    for _ in workers:
        data_queue.put(None)

    for p in workers:
        p.join()


if __name__ == "__main__":
    main()
