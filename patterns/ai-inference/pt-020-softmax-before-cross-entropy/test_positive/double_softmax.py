import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_step(model, inputs, targets, optimizer):
    optimizer.zero_grad()

    logits = model(inputs)
    probs = F.softmax(logits, dim=1)
    loss = F.cross_entropy(probs, targets)

    loss.backward()
    optimizer.step()
    return loss.item()


def compute_loss(model, batch):
    inputs, labels = batch
    outputs = model(inputs)
    soft_outputs = outputs.softmax(dim=-1)
    criterion = nn.CrossEntropyLoss()
    return criterion(soft_outputs, labels)


def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        loss = train_step(model, inputs, targets, optimizer)
        total_loss += loss
    return total_loss / len(dataloader)
