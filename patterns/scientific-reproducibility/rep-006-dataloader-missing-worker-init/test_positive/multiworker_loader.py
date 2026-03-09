import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return np.random.randn(10)


def create_dataloader(dataset, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)


def get_train_loader(data, labels):
    dataset = torch.utils.data.TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=64, num_workers=8)
