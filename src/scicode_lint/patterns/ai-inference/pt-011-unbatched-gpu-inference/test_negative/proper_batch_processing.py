import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class VGGNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            predictions = logits.argmax(dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total
    return accuracy


def predict_all(model, test_data, batch_size=32):
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    test_tensor = torch.from_numpy(test_data).float()
    dataset = TensorDataset(test_tensor)
    loader = DataLoader(dataset, batch_size=batch_size)

    predictions = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            batch_preds = model(batch)
            predictions.append(batch_preds.cpu())

    return torch.cat(predictions, dim=0)
