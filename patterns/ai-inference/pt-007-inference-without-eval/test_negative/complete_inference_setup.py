import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        embedded = self.dropout1(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        pooled = lstm_out.mean(dim=1)
        dropped = self.dropout2(pooled)
        return self.fc(dropped)


def evaluate_test_set(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_input, batch_labels in test_loader:
            logits = model(batch_input)
            preds = logits.argmax(dim=1)
            all_predictions.extend(preds.cpu().tolist())
            all_labels.extend(batch_labels.cpu().tolist())

    return all_predictions, all_labels


def compute_test_accuracy(model, test_data, test_labels):
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == test_labels).float().mean().item()
    return accuracy
