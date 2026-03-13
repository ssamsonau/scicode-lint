import torch
import torch.nn as nn
import torch.nn.functional as F


class BadClassifierLogSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, target):
        logits = self.fc(x)
        log_probs = F.log_softmax(logits, dim=1)
        loss = self.criterion(log_probs, target)
        return loss


def compute_loss_with_log_softmax(model, x, target):
    output = model(x)
    log_probs = output.log_softmax(dim=-1)
    loss = F.cross_entropy(log_probs, target)
    return loss
