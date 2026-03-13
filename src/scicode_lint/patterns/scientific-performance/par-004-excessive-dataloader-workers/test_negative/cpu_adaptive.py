import os

from torch.utils.data import DataLoader


def create_adaptive_loader(dataset, batch_size=32):
    cpu_count = os.cpu_count() or 4
    num_workers = min(4, cpu_count // 2)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
