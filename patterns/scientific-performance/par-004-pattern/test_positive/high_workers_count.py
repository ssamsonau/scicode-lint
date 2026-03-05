import multiprocessing

import torch
from torch.utils.data import DataLoader, TensorDataset


def create_dataloader(data, labels):
    dataset = TensorDataset(data, labels)
    num_cores = multiprocessing.cpu_count()
    loader = DataLoader(dataset, batch_size=64, num_workers=num_cores * 2, pin_memory=True)
    return loader


X = torch.randn(50000, 128)
y = torch.randint(0, 5, (50000,))
train_loader = create_dataloader(X, y)
