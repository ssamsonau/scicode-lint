import random

import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


def generate_dataset():
    features = np.random.randn(1000, 20)
    noise = np.random.normal(0, 0.1, 1000)
    targets = features.sum(axis=1) + noise

    indices = list(range(1000))
    random.shuffle(indices)

    return features[indices], targets[indices]


def build_model():
    layers = []
    for i in range(3):
        layers.append(torch.nn.Linear(20 if i == 0 else 64, 64))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(64, 1))

    return torch.nn.Sequential(*layers)


X, y = generate_dataset()
model = build_model()
