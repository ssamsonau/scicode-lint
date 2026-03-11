import torch
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
        return img, 0


def get_training_loader(train_data, batch_size=64, num_workers=4):
    return DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers
    )


def train_model(model, train_paths, epochs=10):
    dataset = ImageDataset(train_paths)
    loader = get_training_loader(dataset, batch_size=32, num_workers=4)

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        for images, labels in loader:
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
