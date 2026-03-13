import torch.nn as nn
import torch.optim as optim


class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(10000, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 10000)

    def forward(self, src):
        embedded = self.embedding(src)
        transformed = self.transformer(embedded)
        return self.fc(transformed)


def train_model(model, train_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 10000), labels.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return model
