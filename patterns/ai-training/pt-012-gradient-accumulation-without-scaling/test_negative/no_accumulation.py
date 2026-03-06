import torch.nn as nn


def standard_training_loop(model, dataloader, optimizer):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for data, target in dataloader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for batch in loader:
        inputs, labels = batch
        optimizer.zero_grad()

        predictions = model(inputs)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
