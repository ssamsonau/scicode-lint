import numpy as np


def evolve_particle_positions(positions, velocities, dt, box_size):
    pos = positions.copy()
    pos += velocities * dt
    pos %= box_size
    return pos


def apply_decay_and_threshold(concentrations, decay_rate, threshold):
    c = concentrations.copy()
    c *= decay_rate
    c[c < threshold] = 0.0
    c.sort()
    return c


def gaussian_smooth_1d(data, kernel):
    smoothed = data.copy()
    half_k = len(kernel) // 2
    for i in range(half_k, len(smoothed) - half_k):
        smoothed[i] = np.dot(data[i - half_k : i + half_k + 1], kernel)
    smoothed[:half_k] = smoothed[half_k]
    smoothed[-half_k:] = smoothed[-half_k - 1]
    return smoothed
