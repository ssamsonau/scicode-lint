from collections.abc import Callable

import torch
import torch.nn as nn


def create_accumulating_trainer(model: nn.Module, loss_fn: Callable, accumulation_steps: int = 4):
    """Factory function returning a trainer with scaled gradient accumulation."""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    global_step = [0]

    def train_batch(x: torch.Tensor, y: torch.Tensor) -> float:
        model.train()
        pred = model(x)
        batch_loss = loss_fn(pred, y)

        (batch_loss / accumulation_steps).backward()

        global_step[0] += 1
        if global_step[0] % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return batch_loss.item()

    return train_batch


class ScaledAccumulationTrainer:
    """Trainer implementing proper gradient accumulation with loss scaling.

    The key insight: when accumulating gradients over N steps, each mini-batch
    loss must be divided by N before calling backward(). This ensures the
    accumulated gradients have the same magnitude as if we processed a single
    batch of size N * mini_batch_size.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        accumulation_steps: int = 4,
        weight_decay: float = 0.01,
    ):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self._micro_step = 0

    def step(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        self.model.train()
        logits = self.model(inputs)
        loss = self.criterion(logits, targets)

        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()

        self._micro_step += 1
        did_update = False

        if self._micro_step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            did_update = True

        return {"loss": loss.item(), "did_update": did_update}
