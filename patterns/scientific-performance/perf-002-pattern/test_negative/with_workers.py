import torch
from torch.utils.data import DataLoader, TensorDataset


def create_efficient_loader(data, labels):
    dataset = TensorDataset(data, labels)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True, shuffle=True)
    return loader


X_train = torch.randn(50000, 128)
y_train = torch.randint(0, 10, (50000,))
train_loader = create_efficient_loader(X_train, y_train)

for batch_x, batch_y in train_loader:
    pass
