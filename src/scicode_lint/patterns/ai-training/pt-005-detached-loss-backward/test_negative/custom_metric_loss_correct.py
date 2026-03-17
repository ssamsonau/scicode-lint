import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification - keeps gradient connection."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class MultiTaskTrainer:
    """Multi-task trainer with proper gradient flow for all losses."""

    def __init__(self, model: nn.Module, task_weights: list[float]):
        self.model = model
        self.task_weights = task_weights
        self.optimizer = torch.optim.Adam(model.parameters())
        self.losses = [nn.CrossEntropyLoss(), nn.MSELoss()]

    def train_step(self, x: torch.Tensor, targets: list[torch.Tensor]) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(x)

        total_loss = sum(
            w * loss_fn(out, tgt)
            for w, loss_fn, out, tgt in zip(self.task_weights, self.losses, outputs, targets)
        )

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()


def smooth_l1_training_step(model, batch, optimizer, beta=1.0):
    """Training step using smooth L1 loss - gradients flow correctly."""
    inputs, targets = batch
    optimizer.zero_grad()

    predictions = model(inputs)
    loss = F.smooth_l1_loss(predictions, targets, beta=beta)

    loss.backward()
    optimizer.step()

    return loss.item()
