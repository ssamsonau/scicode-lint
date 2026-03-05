import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.images = np.random.randn(num_samples, 3, 224, 224).astype(np.float32)
        self.labels = np.random.randint(0, 10, num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        label = self.labels[idx]
        return image, label


def create_data_loader(dataset, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader


def train_epoch(model, device):
    dataset = ImageDataset(num_samples=5000)
    train_loader = create_data_loader(dataset, batch_size=64)

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
