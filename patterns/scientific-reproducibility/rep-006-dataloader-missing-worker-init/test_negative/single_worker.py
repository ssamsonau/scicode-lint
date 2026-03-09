import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        noise = np.random.randn(*x.shape) * 0.1
        return torch.tensor(x + noise), self.labels[idx]


def create_single_worker_loader(dataset, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)


def create_debug_loader(dataset):
    return DataLoader(dataset, batch_size=16, num_workers=0)


class DataModule:
    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=32, num_workers=0, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=64, num_workers=0)
