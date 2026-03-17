from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class MetricsTracker:
    """Track training metrics using scalar values - no tensor accumulation."""

    window_size: int = 100
    loss_history: deque = field(default_factory=lambda: deque(maxlen=100))
    total_loss: float = 0.0
    step_count: int = 0

    def update(self, loss: torch.Tensor):
        """Record loss as scalar, not tensor."""
        scalar_loss = loss.item()
        self.loss_history.append(scalar_loss)
        self.total_loss += scalar_loss
        self.step_count += 1

    @property
    def running_average(self) -> float:
        if not self.loss_history:
            return 0.0
        return sum(self.loss_history) / len(self.loss_history)

    @property
    def epoch_average(self) -> float:
        if self.step_count == 0:
            return 0.0
        return self.total_loss / self.step_count


class EMALossTracker:
    """Exponential moving average loss tracker using scalars."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.ema: float | None = None

    def update(self, loss: torch.Tensor) -> float:
        scalar = loss.detach().item()
        if self.ema is None:
            self.ema = scalar
        else:
            self.ema = self.alpha * scalar + (1 - self.alpha) * self.ema
        return self.ema


def train_with_callbacks(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    on_batch_end: Callable[[int, float], None] | None = None,
):
    """Training loop with callback receiving scalar loss values."""
    model.train()
    epoch_losses = []

    for batch_idx, (inputs, targets) in enumerate(loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        epoch_losses.append(loss_value)

        if on_batch_end:
            on_batch_end(batch_idx, loss_value)

    return sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
