import multiprocessing as mp

import numpy as np


def train_model_on_batch(batch_data, model_params):
    X, y = batch_data
    weights = np.random.randn(X.shape[1])
    for _ in range(1000):
        predictions = X @ weights
        gradient = X.T @ (predictions - y)
        weights -= 0.001 * gradient
    return weights


def distributed_training(batches, model_params, num_workers=4):
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(
            train_model_on_batch,
            [(batch, model_params) for batch in batches],
        )
    return np.mean(results, axis=0)


batches = [(np.random.randn(1000, 50), np.random.randn(1000)) for _ in range(8)]
final_weights = distributed_training(batches, {})
