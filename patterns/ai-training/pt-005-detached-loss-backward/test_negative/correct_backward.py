import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


def train_step(model, data, targets, optimizer):
    optimizer.zero_grad()

    predictions = model(data)
    loss = nn.functional.mse_loss(predictions, targets)

    loss.backward()
    optimizer.step()

    return loss.item()
