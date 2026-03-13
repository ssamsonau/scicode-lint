import numpy as np


def create_generators(n_experiments, base_seed=42):
    generators = []
    for i in range(n_experiments):
        rng = np.random.default_rng(base_seed + i)
        generators.append(rng)
    return generators


def generate_data(seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((100, 10))


def bootstrap_sample(data, n_samples, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=n_samples)
    return data[indices]


class RandomDataGenerator:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def sample_normal(self, shape):
        return self.rng.standard_normal(shape)

    def sample_uniform(self, low, high, shape):
        return self.rng.uniform(low, high, shape)

    def shuffle(self, array):
        shuffled = array.copy()
        self.rng.shuffle(shuffled)
        return shuffled
