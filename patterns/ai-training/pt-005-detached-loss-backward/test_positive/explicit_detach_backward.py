import torch.nn as nn
import torch.optim as optim


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train_autoencoder(model, data_loader, num_epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in data_loader:
            optimizer.zero_grad()

            reconstruction = model(batch)
            loss = nn.functional.mse_loss(reconstruction, batch)

            detached_loss = loss.detach()
            detached_loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    return model


def train_step(model, inputs, targets, optimizer):
    outputs = model(inputs)
    loss = nn.functional.cross_entropy(outputs, targets)

    safe_loss = loss.clone().detach()
    safe_loss.requires_grad = True
    optimizer.zero_grad()
    safe_loss.backward()
    optimizer.step()

    return safe_loss.item()
