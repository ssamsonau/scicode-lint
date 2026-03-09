import torch
import torch.nn as nn


def train_with_accumulation(model, dataloader, optimizer, accumulation_steps=4):
    model.train()
    for i, (inputs, targets) in enumerate(dataloader):
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


def accumulated_training_loop(model, data, labels, batch_size=8, accum=4):
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(0, len(data), batch_size):
        batch_x = data[i : i + batch_size]
        batch_y = labels[i : i + batch_size]
        loss = nn.functional.mse_loss(model(batch_x), batch_y)
        loss.backward()
        if (i // batch_size + 1) % accum == 0:
            optimizer.step()
            optimizer.zero_grad()
