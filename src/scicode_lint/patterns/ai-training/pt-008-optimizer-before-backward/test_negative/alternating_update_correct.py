from contextlib import contextmanager

import torch
import torch.nn as nn


class OptimizationManager:
    """Manages optimization steps with correct gradient flow."""

    def __init__(self, model: nn.Module, lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler()

    @contextmanager
    def optimization_step(self):
        """Context manager ensuring correct optimization order."""
        self.optimizer.zero_grad()
        try:
            yield
        finally:
            pass

    def step(self, loss: torch.Tensor):
        """Execute optimizer step after backward has been called."""
        loss.backward()
        self.optimizer.step()


def train_discriminator_step(
    discriminator: nn.Module,
    real_batch: torch.Tensor,
    fake_batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Single discriminator training step with correct order."""
    optimizer.zero_grad()

    real_labels = torch.ones(real_batch.size(0), 1)
    fake_labels = torch.zeros(fake_batch.size(0), 1)

    real_loss = criterion(discriminator(real_batch), real_labels)
    fake_loss = criterion(discriminator(fake_batch.detach()), fake_labels)

    total_loss = real_loss + fake_loss
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def train_generator_step(
    generator: nn.Module,
    discriminator: nn.Module,
    noise: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Single generator training step with correct order."""
    optimizer.zero_grad()

    fake_samples = generator(noise)
    fake_labels = torch.ones(fake_samples.size(0), 1)

    loss = criterion(discriminator(fake_samples), fake_labels)
    loss.backward()
    optimizer.step()

    return loss.item()
