"""GAN training with multi-worker DataLoaders missing worker_init_fn."""

import random

import torch
from torch.utils.data import DataLoader, Dataset


class ImagePairDataset(Dataset):
    """Dataset generating random image pairs for contrastive learning."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        anchor = self.images[idx]
        same_class = [i for i, l in enumerate(self.labels) if l == self.labels[idx] and i != idx]
        pos_idx = random.choice(same_class)
        return anchor, self.images[pos_idx], self.labels[idx]


def create_dataloader(images, labels, batch_size=32):
    dataset = ImagePairDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)
