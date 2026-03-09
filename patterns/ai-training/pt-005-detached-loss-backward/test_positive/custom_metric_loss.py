import torch


def train_with_metric(model, data_loader, optimizer, metric_fn):
    model.train()
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        metric = metric_fn(outputs.detach(), targets)
        loss = torch.tensor(metric, requires_grad=True)
        loss.backward()
        optimizer.step()


def compute_loss(pred, target):
    diff = (pred - target).abs().mean()
    return diff.item()
