import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


class AverageMeter:
    """Tracks running average of loss values."""

    def __init__(self):
        self.values = []
        self.sum = torch.tensor(0.0)
        self.count = 0

    def update(self, loss: torch.Tensor, n: int = 1):
        self.values.append(loss)
        self.sum += loss * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0


def train_with_amp_logging(
    model: nn.Module, loader, optimizer, scaler: GradScaler, log_every: int = 10
):
    """AMP training with periodic loss logging."""
    criterion = nn.CrossEntropyLoss()
    batch_losses = []

    for batch_idx, (inputs, targets) in enumerate(loader):
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_losses.append(loss)

        if (batch_idx + 1) % log_every == 0:
            recent_avg = torch.stack(batch_losses[-log_every:]).mean()
            print(f"Step {batch_idx}: loss = {recent_avg:.4f}")

    return batch_losses


class GradientMonitor:
    """Monitors gradient norms during training."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.loss_history = []
        self.grad_norms = []

    def step(self, loss: torch.Tensor, optimizer):
        self.loss_history.append(loss)

        loss.backward()

        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        self.grad_norms.append(total_norm**0.5)

        optimizer.step()
        optimizer.zero_grad()
