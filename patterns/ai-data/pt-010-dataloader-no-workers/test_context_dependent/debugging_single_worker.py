import os

import torch
from torch.utils.data import DataLoader, Dataset


class DebugDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(sample), torch.tensor(label)


def create_debug_loader(dataset, batch_size=8):
    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"

    if debug_mode:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    return loader


def debug_training_step(model, data, labels):
    dataset = DebugDataset(data, labels)
    loader = create_debug_loader(dataset)

    model.train()
    for batch_data, batch_labels in loader:
        output = model(batch_data)
        loss = torch.nn.functional.mse_loss(output, batch_labels)
        loss.backward()
