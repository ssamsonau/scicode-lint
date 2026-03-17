import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class VAETrainer:
    """VAE trainer with gradient accumulation but missing loss scaling."""

    def __init__(self, vae: nn.Module, accumulation_steps: int = 8):
        self.vae = vae
        self.accumulation_steps = accumulation_steps
        self.optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

    def train_epoch(self, loader: DataLoader) -> float:
        self.vae.train()
        total_loss = 0.0

        for batch_idx, (x, _) in enumerate(loader):
            recon, mu, logvar = self.vae(x)

            recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss

            loss.backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()

        return total_loss / len(loader)


def finetune_with_accumulation(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    accumulation: int = 4,
    epochs: int = 5,
):
    """Finetune pretrained model with gradient accumulation - missing scaling."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            inputs, labels = batch["input_ids"], batch["labels"]
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss.backward()

            if (step + 1) % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
