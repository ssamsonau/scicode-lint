import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output(x)


def predict_batch(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        predictions = (logits > 0.5).float()
    return predictions


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            predicted = (outputs > 0.5).int()
            total += batch_labels.size(0)
            correct += (predicted.squeeze() == batch_labels).sum().item()

    accuracy = correct / total
    return accuracy
