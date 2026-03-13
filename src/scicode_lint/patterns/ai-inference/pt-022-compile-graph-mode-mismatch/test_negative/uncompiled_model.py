import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 10)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x.mean(dim=1))


def train_uncompiled(model, train_loader, optimizer, criterion):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()


def validate_uncompiled(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            total_loss += nn.CrossEntropyLoss()(outputs, batch_y).item()
    return total_loss / len(val_loader)


class InferenceEngine:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, inputs):
        with torch.inference_mode():
            return self.model(inputs)
