import time

import torch
import torch.nn as nn


def train_epoch(model, dataloader, optimizer, device):
    start_time = time.time()

    model.train()
    total_loss = 0.0

    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_time = time.time() - start_time
    return total_loss, epoch_time


def process_dataset(model, dataset, device):
    start = time.perf_counter()

    model.eval()
    results = []

    with torch.no_grad():
        for sample in dataset:
            sample = sample.to(device)
            output = model(sample.unsqueeze(0))
            results.append(output.cpu().numpy())

    total_time = time.perf_counter() - start
    return results, total_time


def run_experiment(model, train_loader, val_loader, epochs, device):
    experiment_start = time.time()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        train_loss, _ = train_epoch(model, train_loader, optimizer, device)

    total_experiment_time = time.time() - experiment_start
    return total_experiment_time
