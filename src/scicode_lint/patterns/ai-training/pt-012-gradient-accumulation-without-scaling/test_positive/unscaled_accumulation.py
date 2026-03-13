import torch.nn as nn


def train_with_accumulation(model, dataloader, optimizer, accumulation_steps=4):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(dataloader):
        outputs = model(data)
        loss = criterion(outputs, target)

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


def accumulated_training_epoch(model, loader, optimizer, num_accumulation=8):
    model.train()
    loss_fn = nn.MSELoss()

    optimizer.zero_grad()
    for i, (x, y) in enumerate(loader):
        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()

        if (i + 1) % num_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
