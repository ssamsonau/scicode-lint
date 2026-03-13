import torch


def load_ensemble(paths):
    models = []
    for path in paths:
        state = torch.load(path)
        models.append(state)
    return models


def restore_optimizer(checkpoint_path, optimizer):
    ckpt = torch.load(checkpoint_path)
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"]
