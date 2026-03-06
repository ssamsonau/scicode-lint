import random

import numpy as np
import torch

SEED = 2024

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def initialize_experiment():
    config = {"batch_size": 32, "learning_rate": 0.001, "epochs": 100}

    train_data = np.random.uniform(-1, 1, (5000, 10))
    val_data = np.random.uniform(-1, 1, (1000, 10))

    network = torch.nn.Sequential(
        torch.nn.Linear(10, 128),
        torch.nn.Dropout(0.2),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )

    sample_indices = random.sample(range(5000), 500)

    return config, train_data, val_data, network, sample_indices


result = initialize_experiment()
