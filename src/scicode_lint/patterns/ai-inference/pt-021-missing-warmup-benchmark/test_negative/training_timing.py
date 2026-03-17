"""Training progress timing - operational, not benchmarking."""

import time


def train_epoch(model, dataloader, optimizer, criterion):
    """Train one epoch with timing for progress monitoring."""
    model.train()
    epoch_start = time.time()

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()

        batch_start = time.time()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        batch_time = time.time() - batch_start

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: {batch_time:.3f}s")

    epoch_time = time.time() - epoch_start
    print(f"Epoch completed in {epoch_time:.1f}s")
    return epoch_time
