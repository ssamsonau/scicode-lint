import multiprocessing as mp

import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))

    def forward(self, x):
        return self.layers(x)


def train_worker(rank, data):
    model = Network()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    inputs = torch.tensor(data).to(device)
    outputs = model(inputs)
    print(f"Rank {rank}: {outputs.shape}")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    data_samples = [torch.randn(32, 100).numpy() for _ in range(4)]

    processes = []
    for i, data in enumerate(data_samples):
        p = mp.Process(target=train_worker, args=(i, data))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
