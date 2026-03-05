import multiprocessing

import torch
from torch.utils.data import DataLoader, TensorDataset


def get_optimal_workers():
    num_cores = multiprocessing.cpu_count()
    return min(4, num_cores)


def setup_loader(features, targets):
    dataset = TensorDataset(features, targets)
    loader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=get_optimal_workers(),
        pin_memory=True,
        persistent_workers=True,
    )
    return loader


X_train = torch.randn(100000, 64)
y_train = torch.randint(0, 10, (100000,))
training_loader = setup_loader(X_train, y_train)
