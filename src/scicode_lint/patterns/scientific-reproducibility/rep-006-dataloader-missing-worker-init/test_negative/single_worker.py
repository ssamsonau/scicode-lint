"""DataLoader with num_workers=0 - no worker_init_fn needed."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class AugmentedDataset(Dataset):
    """Dataset with random augmentation."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].copy()
        x += np.random.randn(*x.shape) * 0.1
        return torch.tensor(x)


train_loader = DataLoader(AugmentedDataset(np.random.randn(100, 10)), batch_size=32, num_workers=0)
val_loader = DataLoader(AugmentedDataset(np.random.randn(50, 10)), batch_size=16, num_workers=0)
test_loader = DataLoader(AugmentedDataset(np.random.randn(20, 10)), batch_size=8, num_workers=0)
