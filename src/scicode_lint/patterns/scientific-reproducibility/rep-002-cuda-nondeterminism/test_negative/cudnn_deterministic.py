from contextlib import contextmanager

import torch


@contextmanager
def deterministic_context(seed: int = 42):
    prev_benchmark = torch.backends.cudnn.benchmark
    prev_deterministic = torch.backends.cudnn.deterministic

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = prev_benchmark
        torch.backends.cudnn.deterministic = prev_deterministic


class DeterministicTrainer:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._configure_determinism()

    def _configure_determinism(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    def train_epoch(self, model, dataloader, optimizer, criterion):
        with deterministic_context(self.seed):
            model.train()
            total_loss = 0.0
            for batch in dataloader:
                optimizer.zero_grad()
                loss = criterion(model(batch["input"]), batch["target"])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(dataloader)
