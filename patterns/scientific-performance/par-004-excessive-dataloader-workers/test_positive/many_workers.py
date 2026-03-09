from torch.utils.data import DataLoader


def create_loader(dataset, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, num_workers=64, shuffle=True)


def get_train_loader(train_data):
    return DataLoader(train_data, batch_size=128, num_workers=128, pin_memory=True)
