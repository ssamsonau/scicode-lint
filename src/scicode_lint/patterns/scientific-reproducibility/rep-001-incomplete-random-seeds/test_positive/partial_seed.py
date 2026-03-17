import random

import numpy as np


def train_model(model, data, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    init_weights = [random.gauss(0, 0.1) for _ in range(10)]
    shuffled = data[indices]
    model.coef_ = init_weights
    return model.fit(shuffled)
