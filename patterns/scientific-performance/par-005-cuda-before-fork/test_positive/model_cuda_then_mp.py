from multiprocessing import Pool

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)


def train_on_subset(data_chunk):
    output = model(data_chunk)
    return output.sum().item()


model = SimpleModel()
if torch.cuda.is_available():
    model = model.cuda()

data_chunks = [torch.randn(32, 100) for _ in range(8)]

with Pool(processes=4) as pool:
    results = pool.map(train_on_subset, data_chunks)
