from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image

        img = Image.open(self.paths[idx])
        if self.transform:
            img = self.transform(img)
        return img


def create_data_loader(dataset, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_training_loader(train_data, batch_size=64):
    return DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
