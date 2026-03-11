import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


def train_regression(model, inputs, targets, optimizer):
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = nn.MSELoss()(predictions, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_with_l1_loss(model, inputs, targets, optimizer):
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = nn.L1Loss()(predictions, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


class MultiOutputRegressor:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.SmoothL1Loss()

    def compute_loss(self, inputs, targets):
        outputs = self.model(inputs)
        return self.criterion(outputs, targets)
