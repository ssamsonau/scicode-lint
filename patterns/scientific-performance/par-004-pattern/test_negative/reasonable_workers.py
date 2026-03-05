import torch
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randint(0, 1000, (size, 128))
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


dataset = TextDataset(20000)
loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)

for batch_data, batch_labels in loader:
    pass
