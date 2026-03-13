import torch.nn as nn


def training_loop(model, train_loader, optimizer, accumulation_steps=4):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        for step, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()


class Trainer:
    def __init__(self, model, optimizer, accum_steps=2):
        self.model = model
        self.optimizer = optimizer
        self.accum_steps = accum_steps
        self.criterion = nn.BCEWithLogitsLoss()

    def train_batch(self, batches):
        self.optimizer.zero_grad()

        for i, (x, y) in enumerate(batches):
            logits = self.model(x)
            loss = self.criterion(logits, y)

            loss.backward()

            if (i + 1) % self.accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
