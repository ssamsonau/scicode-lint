import torch
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    def __init__(self, num_samples):
        self.images = torch.randn(num_samples, 3, 224, 224)
        self.labels = torch.randint(0, 100, (num_samples,))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


dataset = ImageDataset(10000)
loader = DataLoader(dataset, batch_size=32, num_workers=6, pin_memory=True, prefetch_factor=2)

for images, labels in loader:
    pass
