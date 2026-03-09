import numpy as np
import torch
from sklearn.model_selection import train_test_split


def setup_experiment():
    np.random.seed(42)


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2)


def initialize_model():
    torch.manual_seed(123)
    return torch.nn.Linear(10, 1)
