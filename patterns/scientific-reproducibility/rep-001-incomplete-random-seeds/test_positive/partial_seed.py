import random

import numpy as np


def train_model(model, data, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    shuffled = data[indices]
    return model.fit(shuffled)


def initialize_weights(model, seed=0):
    random.seed(seed)
    for layer in model.layers:
        layer.weights = [random.gauss(0, 0.1) for _ in range(layer.size)]
