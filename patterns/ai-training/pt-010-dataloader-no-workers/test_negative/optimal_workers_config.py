import torch
from torch.utils.data import DataLoader, TensorDataset


def train_model(model, device):
    dataset = TensorDataset(torch.randn(1000, 3, 224, 224), torch.randint(0, 10, (1000,)))
    loader = DataLoader(dataset, batch_size=32, num_workers=4)

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
