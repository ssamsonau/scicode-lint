import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)


def train_model():
    data = np.random.randn(100, 10)
    labels = np.random.randint(0, 2, 100)

    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(10):
        inputs = torch.from_numpy(data).float()
        targets = torch.from_numpy(labels).long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

    return model


if __name__ == "__main__":
    model = train_model()
