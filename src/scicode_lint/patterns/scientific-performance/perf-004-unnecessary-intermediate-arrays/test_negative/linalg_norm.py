import numpy as np


def compute_distances_and_angles(points, origin):
    diff = points - origin
    distances = np.linalg.norm(diff, axis=1)
    angles = np.arctan2(diff[:, 1], diff[:, 0])
    return distances, angles


def batch_normalize(features):
    mean = features.mean(axis=0)
    std = features.std(axis=0).clip(min=1e-8)
    return (features - mean) / std, mean, std


observations = np.random.randn(5000, 20)
origin = observations.mean(axis=0)
dists, angs = compute_distances_and_angles(observations, origin)
