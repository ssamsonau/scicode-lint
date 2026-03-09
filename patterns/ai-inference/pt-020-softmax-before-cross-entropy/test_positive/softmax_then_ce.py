import torch
import torch.nn as nn
import torch.nn.functional as F


class BadClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, target):
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        loss = self.criterion(probs, target)
        return loss


def compute_loss(model, x, target):
    output = model(x)
    probs = output.softmax(dim=-1)
    loss = F.cross_entropy(probs, target)
    return loss


def train_step(model, batch, labels):
    logits = model(batch)
    softmax_output = torch.softmax(logits, dim=1)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(softmax_output, labels)
    return loss
