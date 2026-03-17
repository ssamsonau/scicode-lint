import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.layers(x)


class DeepClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, n_blocks=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = F.relu(self.input_proj(x))
        h = self.blocks(h)
        return self.head(h)


def compute_loss(model, batch_x, batch_y):
    logits = model(batch_x)
    probs = F.softmax(logits, dim=-1)
    return F.cross_entropy(probs, batch_y)
