import numpy as np


def normalize_per_sample(data):
    means = data.mean(axis=1, keepdims=True)
    stds = data.std(axis=1, keepdims=True)
    normalized = (data - means) / (stds + 1e-8)
    return normalized


def scale_channels(images, channel_scales):
    assert len(channel_scales) == images.shape[-1]
    scaled = images * channel_scales.reshape(1, 1, 1, -1)
    return scaled


batch = np.random.rand(32, 128)
normed = normalize_per_sample(batch)

imgs = np.random.rand(10, 64, 64, 3)
scales = np.array([0.5, 1.0, 0.7])
scaled_imgs = scale_channels(imgs, scales)
