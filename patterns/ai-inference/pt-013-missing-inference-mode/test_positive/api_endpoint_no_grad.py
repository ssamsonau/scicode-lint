import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).mean(dim=1)
        return self.fc(x)


classifier = TextClassifier(10000, 128, 5)
classifier.eval()


def classify_text(token_ids):
    with torch.no_grad():
        logits = classifier(token_ids)
        return torch.softmax(logits, dim=-1)


def get_embeddings(token_ids):
    with torch.no_grad():
        return classifier.embedding(token_ids)
