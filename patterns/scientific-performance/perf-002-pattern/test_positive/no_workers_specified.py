import torch
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 256)
        self.targets = torch.randint(0, 5, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


dataset = CustomDataset(20000)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

for inputs, targets in loader:
    pass
