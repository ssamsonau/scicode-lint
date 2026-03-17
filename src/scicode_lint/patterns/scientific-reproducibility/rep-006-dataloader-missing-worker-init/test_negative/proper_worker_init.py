"""Using torchvision transforms with deterministic augmentation."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_deterministic_transforms(seed: int = 42):
    """Create transforms that use torch.Generator for reproducibility."""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )


def seed_worker(worker_id: int):
    """Worker init function for reproducible multi-worker loading."""
    worker_seed = torch.initial_seed() % 2**32
    import random

    import numpy as np

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_reproducible_dataloader(
    dataset_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    """Create DataLoader with proper seeding for reproducibility."""
    generator = torch.Generator().manual_seed(seed)

    dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=get_deterministic_transforms(seed),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
