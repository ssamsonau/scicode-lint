import numpy as np


def compute_entropy(probabilities):
    eps = 1e-10
    log_probs = np.log(probabilities + eps)
    entropy = -np.sum(probabilities * log_probs)
    return entropy


def log_transform(data):
    safe_data = np.maximum(data, 1e-8)
    transformed = np.log(safe_data)
    return transformed


def compute_log_likelihood(observations):
    log_obs = np.log(np.clip(observations, 1e-10, None))
    likelihood = np.sum(log_obs)
    return likelihood


probs = np.array([0.5, 0.3, 0.2, 0.0])
ent = compute_entropy(probs)

values = np.array([1.0, 2.0, 0.0, 3.0])
result = log_transform(values)
