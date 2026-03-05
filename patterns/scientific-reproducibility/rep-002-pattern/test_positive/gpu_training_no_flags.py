import torch
import torch.nn as nn
import torch.optim as optim


def train_on_gpu():
    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")

    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        inputs = torch.randn(32, 100).to(device)
        labels = torch.randint(0, 10, (32,)).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return model


trained_model = train_on_gpu()
