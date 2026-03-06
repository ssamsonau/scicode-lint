import torch
from torch.utils.data import DataLoader, Dataset


class SensorDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.LongTensor([self.labels[idx]])
        return sequence, label


def build_data_loaders(train_sequences, train_labels, val_sequences, val_labels):
    train_dataset = SensorDataset(train_sequences, train_labels)
    val_dataset = SensorDataset(val_sequences, val_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader


def run_training(model, device, train_sequences, train_labels, val_sequences, val_labels):
    train_loader, val_loader = build_data_loaders(
        train_sequences, train_labels, val_sequences, val_labels
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(100):
        model.train()
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.squeeze().to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
