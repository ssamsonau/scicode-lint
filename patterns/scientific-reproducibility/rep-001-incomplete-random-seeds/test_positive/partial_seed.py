import random

import numpy as np


def train_model(model, data, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    weights = [random.gauss(0, 0.1) for _ in range(10)]
    shuffled = data[indices]
    return model.fit(shuffled)
