"""DataLoader with workers but only deterministic transforms - no worker_init_fn needed."""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DeterministicTransformDataset(Dataset):
    """Dataset with ONLY deterministic transforms - no random operations."""

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        x = self.transform(x)
        return x, self.labels[idx]


def create_deterministic_loader(data, labels, batch_size=32, num_workers=4):
    """Create DataLoader - no worker_init_fn needed since no random ops in Dataset."""
    dataset = DeterministicTransformDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
