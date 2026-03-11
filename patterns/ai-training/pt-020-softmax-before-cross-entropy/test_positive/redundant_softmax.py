import torch.nn as nn
import torch.nn.functional as F


def compute_loss(logits, targets):
    probs = F.softmax(logits, dim=-1)
    return F.cross_entropy(probs, targets)


class Model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, targets):
        logits = self.fc(x)
        probs = self.softmax(logits)
        return nn.CrossEntropyLoss()(probs, targets)
