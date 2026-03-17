import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))


def predict_sentiment(model, token_ids):
    model.eval()
    logits = model(token_ids)
    return torch.softmax(logits, dim=-1)


def evaluate_corpus(model, dataloader):
    model.eval()
    all_preds = []
    for batch_tokens, _ in dataloader:
        preds = model(batch_tokens).argmax(dim=1)
        all_preds.extend(preds.tolist())
    return all_preds
