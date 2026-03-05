import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return torch.sigmoid(self.layer3(x))


def training_step(model, inputs, labels, optimizer):
    predictions = model(inputs)
    loss = nn.functional.binary_cross_entropy(predictions, labels)

    # BUG: Converting to Python scalar before backward
    scalar_loss = float(loss)

    optimizer.zero_grad()
    loss.backward()  # This works but pattern is suspicious
    optimizer.step()

    return scalar_loss
