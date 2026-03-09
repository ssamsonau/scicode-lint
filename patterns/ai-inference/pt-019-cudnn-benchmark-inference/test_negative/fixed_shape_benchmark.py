import torch.backends.cudnn as cudnn


def setup_training():
    cudnn.benchmark = True


def train_fixed_batch(model, dataloader):
    cudnn.benchmark = True
    for batch in dataloader:
        model(batch)
