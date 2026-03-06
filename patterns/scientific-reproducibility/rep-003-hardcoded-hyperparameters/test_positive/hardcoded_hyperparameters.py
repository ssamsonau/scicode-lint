import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


model = NeuralNet()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(50):
    batch = torch.randn(64, 784)
    labels = torch.randint(0, 10, (64,))

    optimizer.zero_grad()
    outputs = model(batch)
    loss = nn.functional.cross_entropy(outputs, labels)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Loss: {loss.item():.4f}")
