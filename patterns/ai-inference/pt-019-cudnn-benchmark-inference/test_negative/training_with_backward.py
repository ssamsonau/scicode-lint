import torch
import torch.backends.cudnn as cudnn


def train_model(model, dataloader, optimizer, criterion):
    cudnn.benchmark = True
    model.train()

    for epoch in range(10):
        for batch, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()


class Trainer:
    def __init__(self, model, optimizer, criterion):
        torch.backends.cudnn.benchmark = True
        self.model = model.cuda()
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(self, dataloader):
        """Run one training epoch with gradient computation."""
        self.model.train()
        for inputs, targets in dataloader:
            self.optimizer.zero_grad()
            predictions = self.model(inputs.cuda())
            loss = self.criterion(predictions, targets.cuda())
            loss.backward()
            self.optimizer.step()
