import torch


def normalize_train_val_combined(train_data, val_data, test_data):
    train_val_combined = torch.cat([train_data, val_data])
    mean = train_val_combined.mean()
    std = train_val_combined.std()
    train_normalized = (train_data - mean) / std
    val_normalized = (val_data - mean) / std
    test_normalized = (test_data - mean) / std
    return train_normalized, val_normalized, test_normalized
