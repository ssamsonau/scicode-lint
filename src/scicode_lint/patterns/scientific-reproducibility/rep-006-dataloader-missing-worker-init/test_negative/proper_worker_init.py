import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def worker_init_fn(worker_id):
    """Seed each worker uniquely based on worker_id."""
    np.random.seed(worker_id + 42)
    torch.manual_seed(worker_id + 42)


class AugmentedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx].copy()
        # Random augmentation
        item += np.random.randn(*item.shape) * 0.1
        return torch.tensor(item)


def create_dataloader(dataset, batch_size=32, num_workers=4):
    """DataLoader with proper worker_init_fn for reproducible augmentation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        shuffle=True,
    )


def create_seeded_loader(dataset, seed=42):
    """DataLoader with worker init that ensures unique seeding per worker."""

    def seed_worker(worker_id):
        np.random.seed(seed + worker_id)

    return DataLoader(dataset, num_workers=8, worker_init_fn=seed_worker)
