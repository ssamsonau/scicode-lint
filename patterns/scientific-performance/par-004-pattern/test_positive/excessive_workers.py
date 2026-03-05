import torch
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 3, 224, 224)
        self.labels = torch.randint(0, 10, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


dataset = ImageDataset(10000)
loader = DataLoader(dataset, batch_size=32, num_workers=32, shuffle=True)

for batch_data, batch_labels in loader:
    pass
