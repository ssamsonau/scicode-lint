import multiprocessing as mp
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass
class LoaderConfig:
    """Configuration for DataLoader with optimal worker settings."""

    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = False

    @classmethod
    def auto_workers(cls, batch_size: int = 32):
        """Create config with auto-detected worker count."""
        cpu_count = mp.cpu_count()
        optimal_workers = min(cpu_count - 1, 8) if cpu_count > 1 else 0
        return cls(
            batch_size=batch_size,
            num_workers=optimal_workers,
            persistent_workers=optimal_workers > 0,
        )


def build_loader(dataset, config: LoaderConfig) -> DataLoader:
    """Build DataLoader from configuration."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        drop_last=config.drop_last,
    )


def get_distributed_loader(dataset, world_size: int, rank: int):
    """Create DataLoader for distributed training with proper workers."""
    from torch.utils.data.distributed import DistributedSampler

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    return DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
