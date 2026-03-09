import torch.nn as nn
import torch.nn.functional as F


class LogSoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        logits = self.fc(x)
        return self.log_softmax(logits)


def train_with_nll(model, inputs, targets, optimizer):
    optimizer.zero_grad()

    log_probs = model(inputs)
    loss = F.nll_loss(log_probs, targets)

    loss.backward()
    optimizer.step()
    return loss.item()


def compute_nll_loss(logits, targets):
    log_probs = F.log_softmax(logits, dim=1)
    return F.nll_loss(log_probs, targets)


class NLLTrainer:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.NLLLoss()

    def step(self, inputs, targets, optimizer):
        optimizer.zero_grad()
        log_probs = F.log_softmax(self.model(inputs), dim=1)
        loss = self.criterion(log_probs, targets)
        loss.backward()
        optimizer.step()
        return loss
