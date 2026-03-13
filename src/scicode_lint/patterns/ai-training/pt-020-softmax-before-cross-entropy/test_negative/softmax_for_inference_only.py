import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads), num_layers=2
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits

    def predict_probs(self, x):
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs


def compute_ce_loss(model, inputs, targets, label_smoothing=0.1):
    logits = model(inputs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return criterion(logits, targets)


def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            logits = model(inputs)
            _loss = F.cross_entropy(logits, labels, reduction="mean")
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total
