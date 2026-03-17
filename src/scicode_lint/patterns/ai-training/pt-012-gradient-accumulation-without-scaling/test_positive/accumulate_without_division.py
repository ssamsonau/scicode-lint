import torch.nn as nn


class MultiTaskHead(nn.Module):
    def __init__(self, backbone, task_dims):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList([nn.Linear(256, d) for d in task_dims])

    def forward(self, x):
        features = self.backbone(x)
        return [head(features) for head in self.heads]


def train_multitask(model, dataloader, optimizer, accum_steps=8):
    model.train()
    task_losses = [nn.MSELoss(), nn.CrossEntropyLoss()]
    optimizer.zero_grad()

    for step, (inputs, targets_list) in enumerate(dataloader):
        outputs = model(inputs)
        total = sum(
            loss_fn(out, tgt) for loss_fn, out, tgt in zip(task_losses, outputs, targets_list)
        )
        total.backward()

        if (step + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


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
