import numpy as np
from torch.utils.data import DataLoader, Dataset


class AugmentedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        noise = np.random.randn(*item.shape) * 0.1
        return item + noise


def create_dataloader(dataset, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def get_train_loader(train_data):
    dataset = AugmentedDataset(train_data)
    loader = DataLoader(dataset, batch_size=64, num_workers=8, shuffle=True)
    return loader
