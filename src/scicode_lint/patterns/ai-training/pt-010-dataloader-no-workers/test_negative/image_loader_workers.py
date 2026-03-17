import os
from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset


class StreamingImageDataset(IterableDataset):
    """Memory-efficient streaming dataset for large image collections."""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __iter__(self) -> Iterator:
        from PIL import Image

        for fname in os.listdir(self.root_dir):
            if fname.endswith((".jpg", ".png")):
                img = Image.open(os.path.join(self.root_dir, fname))
                if self.transform:
                    img = self.transform(img)
                yield img, 0


def create_optimized_loader(
    dataset, batch_size: int = 32, num_workers: int = 4, prefetch_factor: int = 2
) -> DataLoader:
    """Create DataLoader with optimal worker configuration."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


class DataLoaderFactory:
    """Factory for creating DataLoaders with proper worker settings."""

    @staticmethod
    def for_training(dataset, batch_size=64) -> DataLoader:
        workers = min(8, os.cpu_count() or 4)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )

    @staticmethod
    def for_validation(dataset, batch_size=128) -> DataLoader:
        workers = min(4, os.cpu_count() or 2)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        )
