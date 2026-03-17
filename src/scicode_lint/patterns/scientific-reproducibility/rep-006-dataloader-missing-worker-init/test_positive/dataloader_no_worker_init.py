"""Training loop with multi-worker DataLoader missing worker_init_fn."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SpectrogramDataset(Dataset):
    """Audio spectrogram dataset with random time masking."""

    def __init__(self, spectrograms, labels):
        self.spectrograms = spectrograms
        self.labels = labels

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        spec = self.spectrograms[idx].copy()
        mask_start = np.random.randint(0, spec.shape[1] - 10)
        spec[:, mask_start : mask_start + 10] = 0
        return torch.tensor(spec, dtype=torch.float32), self.labels[idx]


def create_dataloader(spectrograms, labels, batch_size=16):
    dataset = SpectrogramDataset(spectrograms, labels)
    return DataLoader(dataset, batch_size=batch_size, num_workers=6, shuffle=True)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
