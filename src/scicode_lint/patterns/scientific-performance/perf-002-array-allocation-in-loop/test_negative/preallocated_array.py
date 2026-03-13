import numpy as np


def process_batches_efficient(data, batch_size):
    n_batches = (len(data) + batch_size - 1) // batch_size
    results = np.zeros((n_batches, batch_size))
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(data))
        results[i, : end - start] = data[start:end] * 2
    return results


def compute_frame_features_efficient(video_frames, n_features=128):
    n_frames = len(video_frames)
    all_features = np.zeros((n_frames, n_features))
    for i, frame in enumerate(video_frames):
        all_features[i, :64] = frame.mean(axis=0)
        all_features[i, 64:] = frame.std(axis=0)
    return all_features


def simulate_particles_efficient(n_steps, n_particles):
    trajectories = np.zeros((n_steps, n_particles, 3))
    for step in range(n_steps):
        trajectories[step] = np.random.randn(n_particles, 3)
    return trajectories
