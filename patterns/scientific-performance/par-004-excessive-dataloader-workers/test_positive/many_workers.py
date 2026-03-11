from torch.utils.data import DataLoader


def create_loader(dataset, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, num_workers=64, shuffle=True)
