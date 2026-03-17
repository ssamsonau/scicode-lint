"""DataLoader utilities for various deployment scenarios."""

import platform
from collections.abc import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class DebugDataLoader:
    """DataLoader wrapper with pdb compatibility."""

    def __init__(self, dataset: Dataset, batch_size: int = 32):
        self._loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
        )

    def __iter__(self) -> Iterator:
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)


def create_platform_aware_loader(dataset: Dataset, batch_size: int = 64) -> DataLoader:
    """Create DataLoader with platform-appropriate worker settings."""
    if platform.system() == "Windows":
        return DataLoader(dataset, batch_size=batch_size, num_workers=0)
    else:
        return DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)


class TinyDatasetEvaluator:
    """Evaluator for small datasets with configurable worker usage."""

    def __init__(self, model: nn.Module, threshold: int = 100):
        self.model = model
        self.threshold = threshold

    def evaluate(self, dataset: Dataset) -> float:
        use_workers = len(dataset) > self.threshold
        loader = DataLoader(
            dataset,
            batch_size=min(32, len(dataset)),
            num_workers=4 if use_workers else 0,
        )

        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0.0
