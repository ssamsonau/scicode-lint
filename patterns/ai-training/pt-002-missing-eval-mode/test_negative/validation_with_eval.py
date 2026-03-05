import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        return self.fc(hidden)


def validate(model, val_loader, criterion):
    model.eval()

    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for sequences, labels in val_loader:
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            total_loss += loss.item()

            _, predicted_classes = torch.max(predictions, 1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions

    return avg_loss, accuracy
