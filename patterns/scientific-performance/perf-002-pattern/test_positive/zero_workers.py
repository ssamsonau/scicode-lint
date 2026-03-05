import torch
from torch.utils.data import DataLoader, TensorDataset


def setup_training_loader(features, labels):
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=True)
    return loader


X = torch.randn(50000, 128)
y = torch.randint(0, 10, (50000,))
train_loader = setup_training_loader(X, y)

for batch_x, batch_y in train_loader:
    pass
