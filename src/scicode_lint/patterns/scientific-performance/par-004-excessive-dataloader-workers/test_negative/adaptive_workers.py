import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class AugmentedImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.files = [f for f in os.listdir(image_dir) if f.endswith(".pt")]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tensor = torch.load(os.path.join(self.image_dir, self.files[idx]))
        if self.transform:
            tensor = self.transform(tensor)
        return tensor


class AugmentedLoader:
    def __init__(self, image_dir, batch_size=32):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        available = os.cpu_count() or 4
        num_workers = min(8, max(2, available // 2))
        self.train_dataset = AugmentedImageDataset(image_dir, self.transform)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
        )
        self.val_dataset = AugmentedImageDataset(image_dir)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size * 2,
            num_workers=num_workers,
            shuffle=False,
        )
