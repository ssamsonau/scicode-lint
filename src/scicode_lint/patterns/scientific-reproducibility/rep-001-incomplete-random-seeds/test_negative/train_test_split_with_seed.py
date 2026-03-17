import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def set_all_seeds(seed: int = 42):
    """Set seeds for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_split_data(filepath: str, seed: int = 42):
    """Load data and split with complete seed coverage."""
    set_all_seeds(seed)

    data = np.random.randn(1000, 10)
    labels = np.random.randint(0, 2, 1000)

    return train_test_split(data, labels, test_size=0.2, random_state=seed)
