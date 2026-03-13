import torch
import torch.nn as nn


def set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def train_cnn(model, train_loader, criterion, optimizer, epochs=10):
    set_deterministic()
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


def setup_model():
    set_deterministic()
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3),
        nn.ReLU(),
    ).cuda()
    return model
