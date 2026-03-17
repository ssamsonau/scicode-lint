import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, images, labels):
        feats = self.features(images).flatten(1)
        logits = self.head(feats)
        probs = F.softmax(logits, dim=1)
        return self.loss_fn(probs, labels)


def evaluate_ensemble(models, x, target):
    avg_probs = torch.zeros(x.size(0), 10)
    for m in models:
        logits = m(x)
        avg_probs = avg_probs + logits.softmax(dim=-1)
    avg_probs = avg_probs / len(models)
    return F.cross_entropy(avg_probs, target)
