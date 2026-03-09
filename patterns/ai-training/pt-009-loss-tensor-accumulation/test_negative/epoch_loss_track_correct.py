def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


class Trainer:
    def __init__(self, model):
        self.model = model
        self.losses = []

    def train_batch(self, batch, criterion, optimizer):
        loss = criterion(self.model(batch[0]), batch[1])
        loss.backward()
        optimizer.step()
        self.losses.append(loss.item())
