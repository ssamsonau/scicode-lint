import torch
import torch.nn as nn
import torch.optim as optim


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 5))
        self.bias = nn.Parameter(torch.randn(5))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias


def train_with_inplace_modification(model, data, target):
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    output = model(data)
    loss = ((output - target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()

    # BUG: In-place modification of parameters using +=
    for param in model.parameters():
        noise = torch.randn_like(param) * 0.001
        param.data += noise  # In-place operation breaks autograd

    optimizer.step()

    return loss.item()
