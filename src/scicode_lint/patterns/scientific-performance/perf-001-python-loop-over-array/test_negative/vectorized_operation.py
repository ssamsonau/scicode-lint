import numpy as np


def apply_sigmoid(activations):
    return 1.0 / (1.0 + np.exp(-activations))


def compute_loss(predictions, targets):
    epsilon = 1e-7
    clipped = np.clip(predictions, epsilon, 1.0 - epsilon)
    return -np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped))


layer_output = np.random.randn(256, 128)
probabilities = apply_sigmoid(layer_output)
labels = np.random.randint(0, 2, size=(256, 128)).astype(float)
loss = compute_loss(probabilities, labels)
