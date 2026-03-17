import torch
import torch.nn as nn


class GradientAccumulator:
    """Handles gradient accumulation with proper loss scaling."""

    def __init__(self, model: nn.Module, optimizer, accumulation_steps: int = 4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def accumulate(self, loss: torch.Tensor) -> bool:
        """Accumulate scaled gradients. Returns True when optimizer stepped."""
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()

        self.step_count += 1
        if self.step_count % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            return True
        return False


def train_with_mixed_precision_accumulation(model, loader, optimizer, scaler, accum_steps=8):
    """Gradient accumulation with AMP scaler - loss properly divided."""
    model.train()
    criterion = nn.CrossEntropyLoss()

    for i, (inputs, targets) in enumerate(loader):
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets) / accum_steps
        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


def effective_batch_training(model, dataset, base_batch=16, effective_batch=128):
    """Achieve large effective batch size through properly scaled accumulation."""
    from torch.utils.data import DataLoader

    accumulation = effective_batch // base_batch
    loader = DataLoader(dataset, batch_size=base_batch, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = nn.MSELoss()

    for step, (x, y) in enumerate(loader):
        loss = criterion(model(x), y)
        (loss / accumulation).backward()

        if (step + 1) % accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
