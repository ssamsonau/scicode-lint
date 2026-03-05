import torch
import torch.nn as nn


class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        final_hidden = lstm_out[:, -1, :]
        logits = self.classifier(final_hidden)
        return logits


def classify_text(model, input_tensor):
    model.eval()
    with torch.no_grad():
        raw_output = model(input_tensor)
        probability = torch.sigmoid(raw_output)
        is_positive = probability > 0.5
    return is_positive


def process_reviews(model, review_loader):
    model.eval()
    positive_count = 0
    negative_count = 0

    with torch.no_grad():
        for reviews in review_loader:
            scores = model(reviews)
            probs = torch.sigmoid(scores)
            positive_mask = probs > 0.5
            positive_count += positive_mask.sum().item()
            negative_count += (~positive_mask).sum().item()

    return positive_count, negative_count
