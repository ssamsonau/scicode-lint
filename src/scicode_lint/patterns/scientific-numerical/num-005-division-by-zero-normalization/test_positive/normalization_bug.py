import numpy as np


def batch_norm_forward(x, gamma, beta):
    batch_mean = x.mean(axis=0)
    batch_var = x.var(axis=0)
    batch_std = np.sqrt(batch_var)
    x_hat = (x - batch_mean) / batch_std
    return gamma * x_hat + beta


def layer_norm(x, gamma, beta):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var)
    return gamma * x_norm + beta


def instance_norm(feature_maps):
    mean = feature_maps.mean(axis=(2, 3), keepdims=True)
    std = feature_maps.std(axis=(2, 3), keepdims=True)
    normalized = (feature_maps - mean) / std
    return normalized


batch = np.random.randn(32, 64)
gamma = np.ones(64)
beta = np.zeros(64)
out = batch_norm_forward(batch, gamma, beta)
