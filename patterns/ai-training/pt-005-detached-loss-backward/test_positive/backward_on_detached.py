import torch
import torch.nn as nn
import torch.optim as optim


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def train_iteration(model, data, target):
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    output = model(data)
    loss = nn.functional.mse_loss(output, target)

    # BUG: Creating new tensor without gradient history
    loss_disconnected = torch.tensor(loss.data)

    optimizer.zero_grad()
    loss_disconnected.backward()  # No gradients will flow
    optimizer.step()

    return loss_disconnected.item()
