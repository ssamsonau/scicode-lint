import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def setup_experiment(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_data(X, y, seed=42):
    return train_test_split(X, y, test_size=0.2, random_state=seed)


def initialize_model(seed=123):
    torch.manual_seed(seed)
    return torch.nn.Linear(10, 1)
