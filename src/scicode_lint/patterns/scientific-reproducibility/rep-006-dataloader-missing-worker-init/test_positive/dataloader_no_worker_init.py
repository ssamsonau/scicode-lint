import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class AugmentedDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        noise = np.random.randn(*x.shape) * 0.1
        x = x + noise
        return torch.tensor(x), self.labels[idx]


def create_dataloader(dataset):
    loader = DataLoader(dataset, batch_size=32, num_workers=4)
    return loader


if __name__ == "__main__":
    data = np.random.randn(1000, 10)
    labels = np.random.randint(0, 2, 1000)
    dataset = AugmentedDataset(data, labels)
    loader = create_dataloader(dataset)
