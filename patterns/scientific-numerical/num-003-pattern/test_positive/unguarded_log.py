import numpy as np


def compute_entropy(probabilities):
    log_probs = np.log(probabilities)
    entropy = -np.sum(probabilities * log_probs)
    return entropy


def log_transform(data):
    transformed = np.log(data)
    return transformed


def compute_log_likelihood(observations):
    log_obs = np.log(observations)
    likelihood = np.sum(log_obs)
    return likelihood


probs = np.array([0.5, 0.3, 0.2, 0.0])
ent = compute_entropy(probs)

values = np.array([1.0, 2.0, 0.0, 3.0])
result = log_transform(values)
