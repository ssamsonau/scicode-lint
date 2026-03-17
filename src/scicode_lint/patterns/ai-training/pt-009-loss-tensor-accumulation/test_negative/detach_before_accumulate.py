from collections import deque
from typing import Protocol

import torch
import torch.nn as nn


class LossCallback(Protocol):
    def __call__(self, step: int, loss_value: float) -> None: ...


class ScalarLossTracker:
    """Proper loss tracking using scalar values only."""

    def __init__(self, window_size: int = 100):
        self.window = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def record(self, loss: torch.Tensor):
        value = loss.detach().item()
        self.window.append(value)
        self.total += value
        self.count += 1

    @property
    def running_mean(self) -> float:
        return sum(self.window) / len(self.window) if self.window else 0.0

    @property
    def global_mean(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0


def train_with_proper_logging(
    model: nn.Module, loader, optimizer, criterion: nn.Module, callback: LossCallback | None = None
):
    """Training with correct scalar loss tracking."""
    tracker = ScalarLossTracker()

    for step, (inputs, targets) in enumerate(loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        tracker.record(loss)

        if callback:
            callback(step, loss.item())

    return tracker.global_mean


class EpochMetrics:
    """Collects per-epoch metrics as scalars."""

    def __init__(self):
        self.epoch_losses: list[float] = []
        self.epoch_accuracies: list[float] = []

    def log_batch(self, loss: torch.Tensor, correct: int, total: int):
        self.epoch_losses.append(loss.detach().cpu().item())
        self.epoch_accuracies.append(correct / total if total > 0 else 0.0)

    def summarize(self) -> dict[str, float]:
        return {
            "mean_loss": sum(self.epoch_losses) / len(self.epoch_losses),
            "mean_accuracy": sum(self.epoch_accuracies) / len(self.epoch_accuracies),
        }

    def reset(self):
        self.epoch_losses.clear()
        self.epoch_accuracies.clear()
