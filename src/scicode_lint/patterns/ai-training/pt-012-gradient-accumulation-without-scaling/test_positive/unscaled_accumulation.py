import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def train_ddp_unscaled(
    model: nn.Module, train_loader, optimizer, rank: int, accumulation_steps: int = 4
):
    """Distributed training with gradient accumulation - missing loss scaling.

    Even with DDP, gradient accumulation requires dividing loss by
    accumulation_steps to maintain correct gradient magnitudes.
    """
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(rank)
        targets = targets.to(rank)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


class SequenceModelTrainer:
    """Sequence model trainer with unscaled gradient accumulation."""

    def __init__(self, model: nn.Module, vocab_size: int, accumulation: int = 8, lr: float = 5e-4):
        self.model = model
        self.accumulation = accumulation
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor, step: int):
        self.model.train()

        logits = self.model(input_ids)
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()

        if (step + 1) % self.accumulation == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()
