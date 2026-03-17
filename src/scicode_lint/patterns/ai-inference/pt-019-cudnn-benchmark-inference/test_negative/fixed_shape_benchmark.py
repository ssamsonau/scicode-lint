import torch.backends.cudnn as cudnn


def setup_training():
    cudnn.benchmark = True


def train_fixed_batch(model, dataloader, optimizer, criterion):
    cudnn.benchmark = True
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
