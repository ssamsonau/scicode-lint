def train_with_metric(model, data_loader, optimizer, criterion):
    model.train()
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def compute_loss(pred, target):
    return (pred - target).abs().mean()
