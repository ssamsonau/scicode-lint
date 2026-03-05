import torch
import torch.nn as nn
import torch.optim as optim


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(1, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(64, 1, 2, stride=2)

    def forward(self, x):
        d1 = torch.relu(self.down1(x))
        d2 = torch.relu(self.down2(d1))
        u1 = torch.relu(self.up1(d2))
        return torch.sigmoid(self.up2(u1))


def train_segmentation(model, train_loader, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        losses = []

        for images, masks in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().item())

        _avg_loss = sum(losses) / len(losses)

    return model
