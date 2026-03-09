import numpy as np


def process_batches(data, batch_size):
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        result = np.zeros(batch_size)
        for j, val in enumerate(batch):
            result[j] = val * 2
        results.append(result)
    return results


def compute_frame_features(video_frames):
    all_features = []
    for frame in video_frames:
        features = np.zeros(128)
        features[:64] = frame.mean(axis=0)
        features[64:] = frame.std(axis=0)
        all_features.append(features)
    return np.array(all_features)


def simulate_particles(n_steps, n_particles):
    trajectories = []
    for step in range(n_steps):
        positions = np.zeros((n_particles, 3))
        velocities = np.zeros((n_particles, 3))
        positions[:] = np.random.randn(n_particles, 3)
        velocities[:] = np.random.randn(n_particles, 3)
        trajectories.append(positions)
    return trajectories
