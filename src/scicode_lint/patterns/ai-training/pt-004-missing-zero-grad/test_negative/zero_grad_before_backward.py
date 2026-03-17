import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR


class TrainingStep:
    """Encapsulates a single training step with proper gradient management."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        scheduler: OneCycleLR | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        loss.backward()
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        return loss.item()


def train_with_closure(model: nn.Module, loader, optimizer: Optimizer):
    """Training using optimizer closure pattern with zero_grad inside."""
    criterion = nn.CrossEntropyLoss()

    for inputs, targets in loader:

        def closure():
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            return loss

        optimizer.step(closure)


class AccumulatingTrainer:
    """Gradient accumulation with proper zero_grad timing."""

    def __init__(self, model: nn.Module, accumulation_steps: int = 4):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.optimizer = torch.optim.Adam(model.parameters())
        self.criterion = nn.MSELoss()
        self._step = 0

    def train_batch(self, inputs: torch.Tensor, targets: torch.Tensor):
        if self._step % self.accumulation_steps == 0:
            self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets) / self.accumulation_steps
        loss.backward()

        self._step += 1
        if self._step % self.accumulation_steps == 0:
            self.optimizer.step()

        return loss.item() * self.accumulation_steps
