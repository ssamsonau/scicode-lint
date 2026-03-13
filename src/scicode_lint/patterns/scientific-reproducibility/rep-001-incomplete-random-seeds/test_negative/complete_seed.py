import random

import numpy as np
import torch


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, data, seed=42):
    set_all_seeds(seed)
    indices = np.random.permutation(len(data))
    shuffled = data[indices]
    return model.fit(shuffled)


def initialize_weights(model, seed=0):
    set_all_seeds(seed)
    for layer in model.layers:
        layer.weights = [random.gauss(0, 0.1) for _ in range(layer.size)]
