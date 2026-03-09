import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DeterministicAugmentDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.485], std=[0.229]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])
        x = self.transform(x)
        return x, self.labels[idx]


def create_dataloader(dataset, batch_size=32, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


class PreprocessedDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data["inputs"])

    def __getitem__(self, idx):
        return self.data["inputs"][idx], self.data["labels"][idx]


def load_preprocessed_data(data_path, batch_size=32, num_workers=4):
    dataset = PreprocessedDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
