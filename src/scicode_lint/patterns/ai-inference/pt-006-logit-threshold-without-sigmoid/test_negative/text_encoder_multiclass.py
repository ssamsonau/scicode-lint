import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, tokens):
        embedded = self.embedding(tokens).transpose(1, 2)
        conv_out = F.relu(self.conv(embedded))
        pooled = conv_out.max(dim=2)[0]
        return self.fc(pooled)


def classify_documents(model, token_ids_batch):
    model.eval()
    with torch.no_grad():
        logits = model(token_ids_batch)
        labels = logits.argmax(dim=1)
    return labels


def get_sentiment_distribution(model, documents_loader):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in documents_loader:
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            all_predictions.append(probs)

    return torch.cat(all_predictions, dim=0)
