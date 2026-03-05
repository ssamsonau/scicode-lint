import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


def prepare_dataloaders(train_data, train_labels, val_data, val_labels):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = CustomDataset(train_data, train_labels, transform=transform)
    val_dataset = CustomDataset(val_data, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    return train_loader, val_loader


def training_pipeline(model, optimizer, train_data, train_labels, val_data, val_labels):
    train_loader, val_loader = prepare_dataloaders(train_data, train_labels, val_data, val_labels)

    for epoch in range(50):
        model.train()
        total_loss = 0

        for inputs, targets in train_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()

            optimizer.zero_grad()
            predictions = model(inputs)
            loss = torch.nn.functional.cross_entropy(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
